"""ctrl+e (evict_idle) must never touch a model a request is waiting on.

The ds4 provider has no native unload: `unloadModel` terminates the spawned
ds4-server. Pruning a model out from under a client therefore does not just
drop a warm cache -- it kills the server the request is about to be forwarded
to, and the client errors out (or waits out a full respawn). Idle has to mean
"no request holds it *and* no request is queued for it", including requests
that are parked behind another model's load, and including a ds4 request that
names one of the instance's other aliases (a single-resident provider answers
all of its ids from the one resident model).
"""

import asyncio
import threading
import time

import httpx
import pytest

import main
from abstractions.descriptor import ModelDescriptor
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from scheduling import Scheduler

from test_client_disconnect import FakeClient, StreamResponse, make_scope


class DeadAwareClient(FakeClient):
    """Upstream that refuses connections once the provider has been killed.

    Mirrors what a terminated ds4-server does to a request that is forwarded
    to its port: the connection is refused, and the client errors out.
    """

    def __init__(self, provider, response=None):
        super().__init__(response=response)
        self._provider = provider

    async def send(self, request, stream=False):
        # refuses while no server is live: after a prune until the next spawn
        if not self._provider.loaded_ids:
            raise httpx.ConnectError("connection refused")
        return await super().send(request, stream)


class SpawnProvider(Provider):
    """ds4-shaped: one resident model, and unload means kill the server."""

    _type_id = "ds4"
    single_resident = True

    class Model(BaseModel):
        def memory(self) -> float:
            return 100.0

    def __init__(self, model_ids, mem=100.0, kill_delay=0.0):
        self._endpoint_uri = "http://ds4.example:8000/v1"
        self._mem = mem
        self._kill_delay = kill_delay
        self._descriptors = [ModelDescriptor(m, self) for m in model_ids]
        self.spawned = 0
        self.killed = []
        self.loaded_ids = []
        self._lock = threading.Lock()

    @property
    def endpoint_uri(self):
        return self._endpoint_uri

    def getModelsDescriptors(self):
        return list(self._descriptors)

    def getOAIModels(self):
        return [
            {"id": d.modelId, "object": "model", "created": 1, "owned_by": "ds4"}
            for d in self._descriptors
        ]

    def createModel(self, descriptor, loadOptions):
        m = self.Model(descriptor, loadOptions)
        m._mem = self._mem
        return m

    def loadModel(self, model):
        with self._lock:
            self.spawned += 1
            model._loaded = True
            model._load_state = "ready"
            self.loaded_ids.append(model.descriptor.modelId)

    def unloadModel(self, model):
        if self._kill_delay:
            # process.wait(timeout=10) in the real provider: the kill is not
            # instantaneous, so grants can race it.
            time.sleep(self._kill_delay)
        with self._lock:
            model._loaded = False
            if model.descriptor.modelId in self.loaded_ids:
                self.loaded_ids.remove(model.descriptor.modelId)
            self.killed.append(model.descriptor.modelId)


class SlowProvider(Provider):
    """A second provider whose load blocks, to park the coordinator."""

    _type_id = "slow"
    single_resident = False

    class Model(BaseModel):
        def memory(self) -> float:
            return 100.0

    def __init__(self, model_id, delay=0.6):
        self._endpoint_uri = "http://slow.example/v1"
        self._model_id = model_id
        self._delay = delay
        self._descriptors = [ModelDescriptor(model_id, self)]
        self.killed = []

    @property
    def endpoint_uri(self):
        return self._endpoint_uri

    def getModelsDescriptors(self):
        return list(self._descriptors)

    def getOAIModels(self):
        return [
            {"id": d.modelId, "object": "model", "created": 1, "owned_by": "slow"}
            for d in self._descriptors
        ]

    def createModel(self, descriptor, loadOptions):
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model):
        time.sleep(self._delay)
        model._loaded = True
        model._load_state = "ready"

    def unloadModel(self, model):
        model._loaded = False
        self.killed.append(model.descriptor.modelId)


class Recorder:
    """Runs one ASGI request and records what the app streamed back."""

    def __init__(self, scope):
        self._body = scope.pop("_body")
        self._scope = scope
        self.body = b""
        self.chunks = 0
        self.finished = False
        self.error = None
        self._served_request = False
        self._hung_up = asyncio.Event()

    def hang_up(self):
        """Peer goes away: the pending receive() returns http.disconnect."""
        self._hung_up.set()

    async def receive(self):
        if not self._served_request:
            self._served_request = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        # Starlette watches receive() for the disconnect; block here until the
        # test hangs up (a receive that returns eagerly spins the event loop).
        await self._hung_up.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        if message["type"] == "http.response.body":
            self.body += message.get("body", b"")
            self.chunks += 1
            if not message.get("more_body"):
                self.finished = True

    async def run(self):
        try:
            await main.app(self._scope, self.receive, self.send)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.error = e
        self.finished = True

    async def wait_chunks(self, n):
        for _ in range(300):
            if self.chunks >= n or self.error is not None:
                break
            await asyncio.sleep(0.01)


@pytest.fixture
def install():
    def _install(*providers, response=None):
        main.PROVIDERS = list(providers)
        main.SCHEDULER = Scheduler(main.PROVIDERS, budget_mib=1000)
        client = FakeClient(response=response or StreamResponse())
        main.httpx.AsyncClient = lambda *a, **k: client

    yield _install
    main.PROVIDERS = []
    main.SCHEDULER = None


def inflight(scheduler):
    return {m.descriptor.modelId: n for m, n in scheduler.in_flight.items() if n}


def resident(scheduler):
    return [m.descriptor.modelId for m in scheduler.resident]


def test_prune_spares_a_model_that_is_streaming(install):
    """ctrl+e while a client is mid-generation must not kill its server."""
    ds4 = SpawnProvider(["ds4flash"])
    install(ds4, response=StreamResponse(chunks=[b'data: {"x":1}\n\n'], stall=True))

    async def scenario():
        try:
            await main.SCHEDULER.start()
            rec = Recorder(make_scope("ds4flash"))
            task = asyncio.create_task(rec.run())
            await rec.wait_chunks(2)
            assert rec.chunks >= 2, "request never reached the relay"
            assert inflight(main.SCHEDULER) == {"ds4flash": 1}

            await main.SCHEDULER.evict_idle()  # ctrl+e
            await asyncio.sleep(0.1)
            assert ds4.killed == [], "prune killed a server with a live request"
            assert resident(main.SCHEDULER) == ["ds4flash"]

            await main.SCHEDULER.evict_idle()
            await asyncio.sleep(0.1)
            assert ds4.killed == []
            assert rec.error is None
            assert b'"x":1' in rec.body

            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert ds4.loaded_ids == ["ds4flash"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_prune_spares_a_model_with_a_queued_request(install):
    """A request queued for an idle model makes it non-idle.

    The model is genuinely unclaimed at the moment of the prune -- the claim
    only lands when the coordinator reaches the queue -- so idle has to mean
    "no in-flight claim *and* nothing waiting on it", or ctrl+e kills the
    server a queued request is about to be forwarded to.
    """
    ds4 = SpawnProvider(["ds4flash"])
    slow = SlowProvider("big-slow", delay=0.6)
    install(ds4, slow, response=StreamResponse(chunks=[b'data: {"x":1}\n\n']))

    async def scenario():
        try:
            await main.SCHEDULER.start()
            # bring ds4 up and let that request finish: resident, in_flight 0
            warm = Recorder(make_scope("ds4flash"))
            await warm.run()
            assert warm.error is None
            assert resident(main.SCHEDULER) == ["ds4flash"]
            assert inflight(main.SCHEDULER) == {}

            # park the coordinator in a slow load, then queue ds4 behind it
            slow_task = asyncio.create_task(Recorder(make_scope("big-slow")).run())
            await asyncio.sleep(0.1)
            queued = Recorder(make_scope("ds4flash"))
            queued_task = asyncio.create_task(queued.run())
            await asyncio.sleep(0.1)
            assert len(main.SCHEDULER.pending) == 1, "ds4 request should be queued"
            assert inflight(main.SCHEDULER) == {}

            await main.SCHEDULER.evict_idle()  # ctrl+e
            await asyncio.sleep(0.05)
            assert ds4.killed == [], "prune killed a model a request is queued for"

            await asyncio.wait_for(queued_task, 10)
            await asyncio.wait_for(slow_task, 10)
            assert ds4.spawned == 1, "pruned and respawned a model that was in use"
            assert queued.error is None
            assert b'"x":1' in queued.body
            assert b"provider_start_failed" not in queued.body
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_prune_does_not_race_a_grant_with_a_slow_kill(install):
    """The kill takes time (process.wait); a grant landing mid-kill must not
    forward to a server that is on its way out."""
    ds4 = SpawnProvider(["ds4flash"], kill_delay=0.4)
    slow = SlowProvider("big-slow", delay=0.6)
    install(ds4, slow, response=StreamResponse(chunks=[b'data: {"x":1}\n\n']))

    async def scenario():
        try:
            await main.SCHEDULER.start()
            warm = Recorder(make_scope("ds4flash"))
            await warm.run()
            assert inflight(main.SCHEDULER) == {}

            slow_task = asyncio.create_task(Recorder(make_scope("big-slow")).run())
            await asyncio.sleep(0.1)
            queued = Recorder(make_scope("ds4flash"))
            queued_task = asyncio.create_task(queued.run())
            await asyncio.sleep(0.1)
            # the prune dispatches a slow kill; the grant lands inside it
            prune = asyncio.create_task(main.SCHEDULER.evict_idle())
            await asyncio.sleep(0.1)
            await asyncio.wait_for(queued_task, 10)
            await asyncio.wait_for(prune, 10)
            await asyncio.wait_for(slow_task, 10)

            assert ds4.killed == [], "prune raced a queued request's grant"
            assert queued.error is None
            assert b'"x":1' in queued.body
            assert b"provider_start_failed" not in queued.body
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_prune_never_grants_a_model_it_is_killing(monkeypatch):
    """A request that arrives mid-kill must wait for the teardown, not be
    forwarded to a server that is on its way out."""
    ds4 = SpawnProvider(["ds4flash"], kill_delay=0.4)
    main.PROVIDERS = [ds4]
    main.SCHEDULER = Scheduler([main.PROVIDERS[0]], budget_mib=1000)
    monkeypatch.setattr(main, "STARTUP_ATTEMPTS", 1)
    client = DeadAwareClient(ds4, response=StreamResponse(chunks=[b'data: {"x":1}\n\n']))
    main.httpx.AsyncClient = lambda *a, **k: client

    async def scenario():
        try:
            await main.SCHEDULER.start()
            warm = Recorder(make_scope("ds4flash"))
            await warm.run()
            assert warm.error is None
            assert ds4.spawned == 1

            # the prune has cleared its idle checks and is inside the kill...
            prune = asyncio.create_task(main.SCHEDULER.evict_idle())
            await asyncio.sleep(0.05)
            # ...when a new request for that same model arrives
            late = Recorder(make_scope("ds4flash"))
            await asyncio.wait_for(late.run(), 20)
            await asyncio.wait_for(prune, 20)

            assert b"provider_start_failed" not in late.body, (
                "request was forwarded to a server the prune was killing"
            )
            assert late.error is None
            assert b'"x":1' in late.body
            # exactly one spawn per resident generation: the old one is reaped
            # before the next is started, so the port is never contended.
            assert ds4.spawned == 2
            assert ds4.killed == ["ds4flash"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_prune_spares_a_single_resident_alias(install):
    """A queued request for another id of the same ds4 instance counts too."""
    ds4 = SpawnProvider(["ds4flash", "ds4flash-alias"])
    slow = SlowProvider("big-slow", delay=0.6)
    install(ds4, slow, response=StreamResponse(chunks=[b'data: {"x":1}\n\n']))

    async def scenario():
        try:
            await main.SCHEDULER.start()
            warm = Recorder(make_scope("ds4flash"))
            await warm.run()
            assert resident(main.SCHEDULER) == ["ds4flash"]

            slow_task = asyncio.create_task(Recorder(make_scope("big-slow")).run())
            await asyncio.sleep(0.1)
            # the alias resolves to the same resident model (single_resident)
            queued = Recorder(make_scope("ds4flash-alias"))
            queued_task = asyncio.create_task(queued.run())
            await asyncio.sleep(0.1)
            assert len(main.SCHEDULER.pending) == 1

            await main.SCHEDULER.evict_idle()  # ctrl+e
            await asyncio.sleep(0.05)
            assert ds4.killed == [], (
                "prune ignored a queued request naming another id of the "
                "same single-resident instance"
            )

            await asyncio.wait_for(queued_task, 10)
            await asyncio.wait_for(slow_task, 10)
            assert ds4.spawned == 1
            assert b'"x":1' in queued.body
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_prune_still_evicts_a_genuinely_idle_model(install):
    """The guard must not neuter the manual prune."""
    ds4 = SpawnProvider(["ds4flash"])
    install(ds4)

    async def scenario():
        try:
            await main.SCHEDULER.start()
            rec = Recorder(make_scope("ds4flash"))
            task = asyncio.create_task(rec.run())
            await rec.wait_chunks(2)
            rec_body_during = rec.body
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0.05)
            assert inflight(main.SCHEDULER) == {}

            await main.SCHEDULER.evict_idle()
            assert ds4.killed == ["ds4flash"]
            assert resident(main.SCHEDULER) == []
            assert ds4.loaded_ids == []
            assert b'"x":1' in rec_body_during
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())
