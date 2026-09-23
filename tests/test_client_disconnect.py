"""Client-disconnect safety: an abandoned request must never hold a model.

A VS Code assistant (Twinny and friends) aborts in-flight requests constantly:
Esc, a new keystroke, accepting a completion, or the extension host dying while
a generation is running. YAALLB marks a model in-flight for the duration of a
request and *never* evicts an in-flight model (ctrl+e, the TTL sweep and
budget reallocation all skip it), so a claim that outlives its request makes
that model permanently unevictable.

These tests drive the ASGI app directly so the client can hang up mid-request
the way a real socket peer does: uvicorn advertises ASGI spec 2.3, so
Starlette watches `receive()` for `http.disconnect` and cancels the response
body at the exact await point the request happens to be parked on -- waiting
for a model load, sitting in the startup-retry loop, or relaying SSE bytes.
"""

import asyncio
import json

import httpx
import pytest

import main
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from scheduling import Scheduler


class SlowProvider(Provider):
    """Provider with a slow, observable load and a recorded unload."""

    _type_id = "slow"
    single_resident = False

    class Model(BaseModel):
        def memory(self) -> float:
            return 100.0

    def __init__(self, model_ids, mem=100.0, load_delay=0.0):
        self._endpoint_uri = "http://upstream.example/v1"
        self._mem = mem
        self._load_delay = load_delay
        self._descriptors = [ModelDescriptor(m, self) for m in model_ids]
        self.loaded_ids = []
        self.load_calls = []
        self.unloaded_ids = []

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
        m = self.Model(descriptor, loadOptions)
        m._mem = self._mem
        return m

    def loadModel(self, model):
        import time

        self.load_calls.append(model.descriptor.modelId)
        if self._load_delay:
            time.sleep(self._load_delay)
        model._loaded = True
        self.loaded_ids.append(model.descriptor.modelId)

    def unloadModel(self, model):
        model._loaded = False
        if model.descriptor.modelId in self.loaded_ids:
            self.loaded_ids.remove(model.descriptor.modelId)
        self.unloaded_ids.append(model.descriptor.modelId)


class StreamResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, chunks=None, stall=False):
        self.chunks = chunks or [b'data: {"x":1}\n\n']
        self.stall = stall

    async def aiter_raw(self):
        for chunk in self.chunks:
            yield chunk
        if self.stall:
            # a long generation: the client hangs up while we are parked here
            await asyncio.sleep(3600)


class FakeClient:
    """httpx.AsyncClient stand-in: streams, and remembers if it was closed."""

    def __init__(self, response=None, send_exc=None):
        self.response = response or StreamResponse()
        self.send_exc = send_exc
        self.closed = False

    def build_request(self, method, url, json, headers):
        return {"url": url, "json": json, "headers": headers}

    async def send(self, request, stream=False):
        if self.send_exc is not None:
            raise self.send_exc
        return self.response

    async def aclose(self):
        self.closed = True


def make_scope(model_id="model-a", ctx=None):
    body = {"model": model_id, "stream": True, "messages": []}
    if ctx:
        body["context_length"] = ctx
    return {
        "type": "http",
        # uvicorn advertises 2.3, which is what makes Starlette watch for the
        # disconnect and cancel the body generator. Keep it identical.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "client": ("127.0.0.1", 55555),
        "server": ("127.0.0.1", 4343),
        "headers": [(b"content-type", b"application/json")],
        "_body": json.dumps(body).encode(),
    }


async def drive_disconnect(scope, *, chunks_to_wait=1):
    """POST `scope` and hang up once the response body starts arriving.

    Returns once the app has unwound. Mirrors what uvicorn does on a closed
    socket: the pending `receive()` yields `http.disconnect`, Starlette
    cancels the streaming body, and the request dies wherever it is parked.
    """
    body = scope.pop("_body")
    state = {"served_body": False, "body_seen": 0}
    first = asyncio.Event()

    async def receive():
        if not state["served_body"]:
            state["served_body"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        # Client keeps the connection open until it has seen some bytes, then
        # goes away hard (Esc / extension host exit).
        await first.wait()
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            state["body_seen"] += 1
            if state["body_seen"] >= chunks_to_wait:
                first.set()

    try:
        await main.app(scope, receive, send)
    except BaseException:
        # Starlette surfaces the disconnect as ClientDisconnect/Cancelled; the
        # point of the test is the scheduler state afterwards, not the raise.
        pass
    first.set()


def inflight(scheduler):
    return {m.descriptor.modelId: n for m, n in scheduler.in_flight.items() if n}


@pytest.fixture
def app_with(monkeypatch):
    def _install(provider, response=None, send_exc=None):
        main.PROVIDERS = [provider]
        main.SCHEDULER = Scheduler([provider], budget_mib=1000)
        client = FakeClient(response=response, send_exc=send_exc)
        monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **k: client)
        return client

    yield _install
    main.PROVIDERS = []
    main.SCHEDULER = None


def test_disconnect_during_model_load_releases_model(app_with):
    """The window the prelim SSE exists for: load is slow, client gives up."""
    prov = SlowProvider(["model-a"], load_delay=0.3)
    app_with(prov)

    async def scenario():
        await main.SCHEDULER.start()
        try:
            await drive_disconnect(make_scope("model-a"))
            # the load still finishes on the coordinator
            await asyncio.sleep(0.5)
            assert prov.loaded_ids == ["model-a"]
            assert inflight(main.SCHEDULER) == {}, "abandoned load leaked a claim"

            await main.SCHEDULER.evict_idle()  # this is what ctrl+e calls
            assert prov.unloaded_ids == ["model-a"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_disconnect_while_queued_releases_model(app_with):
    """A request abandoned in the queue must not load, nor leak a claim."""
    prov = SlowProvider(["model-a", "model-b"], load_delay=0.3)
    app_with(prov)

    async def scenario():
        await main.SCHEDULER.start()
        try:
            # A parks the coordinator in its load; B queues behind it.
            await asyncio.gather(
                drive_disconnect(make_scope("model-a")),
                drive_disconnect(make_scope("model-b")),
            )
            await asyncio.sleep(0.5)
            assert inflight(main.SCHEDULER) == {}, "abandoned queue entry leaked"
            # B was abandoned before the coordinator reached it: no pointless
            # multi-second load for a client that is already gone.
            assert prov.load_calls == ["model-a"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_disconnect_during_startup_retry_releases_model(app_with):
    """Provider not answering yet: the retry loop must not keep the model."""
    prov = SlowProvider(["model-a"])
    app_with(prov, send_exc=httpx.ConnectError("connection refused"))

    async def scenario():
        await main.SCHEDULER.start()
        try:
            await drive_disconnect(make_scope("model-a"))
            await asyncio.sleep(0.1)
            assert inflight(main.SCHEDULER) == {}, "abandoned retry loop leaked"
            await main.SCHEDULER.evict_idle()
            assert prov.unloaded_ids == ["model-a"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_disconnect_mid_stream_releases_model(app_with):
    """A generation cut short (Esc mid-reply) releases too."""
    prov = SlowProvider(["model-a"])
    app_with(prov, response=StreamResponse(stall=True))

    async def scenario():
        await main.SCHEDULER.start()
        try:
            # hang up only once the relay is actually under way (prelim event
            # plus the first upstream chunk have both been sent)
            await drive_disconnect(make_scope("model-a"), chunks_to_wait=2)
            await asyncio.sleep(0.1)
            assert inflight(main.SCHEDULER) == {}, "abandoned relay leaked"
            await main.SCHEDULER.evict_idle()
            assert prov.unloaded_ids == ["model-a"]
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())


def test_evict_idle_after_abort_frees_vram(app_with):
    """End-to-end statement of the reported symptom: after the editor goes
    away mid-request, ctrl+e must actually free the model."""
    prov = SlowProvider(["model-a"], load_delay=0.2)
    app_with(prov)

    async def scenario():
        await main.SCHEDULER.start()
        try:
            await drive_disconnect(make_scope("model-a"))
            await asyncio.sleep(0.4)
            assert main.SCHEDULER.resident, "model should still be resident"
            await main.SCHEDULER.evict_idle()
            assert main.SCHEDULER.resident == []
            assert prov.loaded_ids == []
        finally:
            await main.SCHEDULER.stop()

    asyncio.run(scenario())
