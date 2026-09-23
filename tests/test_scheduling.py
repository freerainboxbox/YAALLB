import asyncio
import time

import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from scheduling import (
    ModelNotFound,
    Scheduler,
    _impact_suffix,
    select_evictions,
)


class MemModel(BaseModel):
    def __init__(self, mem: float):
        super().__init__(ModelDescriptor("m", None), LoadOptions())
        self._mem = mem

    def memory(self) -> float:
        return self._mem


class MemProvider(Provider):
    _type_id = "mem"
    single_resident = False

    class Model(BaseModel):
        def memory(self) -> float:
            return self._mem

    def __init__(self, endpoint_uri: str, memories: dict):
        self._endpoint_uri = endpoint_uri
        self._memories = dict(memories)
        self._descriptors = [ModelDescriptor(m, self) for m in self._memories]
        self._loaded = []

    @property
    def endpoint_uri(self) -> str:
        return self._endpoint_uri

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        return list(self._descriptors)

    def createModel(self, descriptor, loadOptions) -> BaseModel:
        model = self.Model(descriptor, loadOptions)
        model._mem = self._memories[descriptor.modelId]
        return model

    def loadModel(self, model: BaseModel) -> None:
        model._loaded = True
        self._loaded.append(model)

    def unloadModel(self, model: BaseModel) -> None:
        model._loaded = False
        if model in self._loaded:
            self._loaded.remove(model)


# ---- select_evictions (pure) ----

def test_evict_single_model_closest():
    models = [MemModel(60), MemModel(120)]
    assert select_evictions(models, 100) == [models[1]]


def test_evict_set_closer_than_single():
    models = [MemModel(60), MemModel(50), MemModel(120)]
    # B greedy: 50+60=110 (over 10); A single 120 (over 20) -> B wins
    assert select_evictions(models, 100) == [models[1], models[0]]


def test_evict_tie_break_single():
    models = [MemModel(100), MemModel(50), MemModel(50)]
    # both free exactly 100 -> tie -> single
    assert select_evictions(models, 100) == [models[0]]


def test_evict_no_solution_raises():
    models = [MemModel(30), MemModel(40)]
    with pytest.raises(RuntimeError):
        select_evictions(models, 100)


def test_evict_excludes_zero_memory():
    models = [MemModel(0), MemModel(120)]
    assert select_evictions(models, 100) == [models[1]]


# ---- VRAM impact log suffix ----


def test_impact_suffix_load():
    # 1000 MiB already loaded, 2000 MiB model incoming -> +2000 (1000 -> 3000).
    assert _impact_suffix(1000, 2000, False) == (
        " +2000 MiB computed impact (1000 -> 3000)"
    )


def test_impact_suffix_evict():
    # Eviction uses a negative impact: -2000 (1000 -> -1000).
    assert _impact_suffix(1000, 2000, True) == (
        " -2000 MiB computed impact (1000 -> -1000)"
    )


def test_impact_suffix_rounds_floats():
    assert _impact_suffix(150.4, 83.7, False) == (
        " +84 MiB computed impact (150 -> 234)"
    )


def test_scheduler_logs_vram_impact(capsys):
    async def scenario():
        p = MemProvider("http://a", {"m1": 100, "m2": 80})
        s = Scheduler([p], budget_mib=150)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            # m2 needs to evict m1: logs an unload (-100) then a load (+80).
            m2 = await s.submit("m2", LoadOptions()); s.release(m2)
        finally:
            await s.stop()

    run(scenario())
    captured = capsys.readouterr().err
    assert "-100 MiB computed impact" in captured
    assert "+80 MiB computed impact" in captured


# ---- safety buffer (vram_mib) ----


def test_vram_mib_adds_provider_safety_buffer():
    p = MemProvider("http://a", {"m1": 100})
    p.safety_buffer_mib = 50
    m = p.createModel(p.getModelsDescriptors()[0], LoadOptions())
    assert m.memory() == 100
    assert m.vram_mib() == 150


def test_vram_mib_no_buffer_defaults_to_memory():
    p = MemProvider("http://a", {"m1": 100})
    m = p.createModel(p.getModelsDescriptors()[0], LoadOptions())
    assert m.vram_mib() == 100


def test_select_evictions_uses_safety_buffer():
    # Raw memory 40+40=80 cannot free a 100-MiB shortfall (RuntimeError), but
    # with a 30-MiB safety buffer each model's vram_mib is 70 -> B frees 140.
    models = [MemModel(40), MemModel(40)]
    for m in models:
        m.descriptor.provider = MemProvider("http://a", {})
        m.descriptor.provider.safety_buffer_mib = 30
    assert select_evictions(models, 100) == [models[0], models[1]]


def test_scheduler_evicts_for_safety_buffer_headroom():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80})
        p.safety_buffer_mib = 40
        # Raw memory 80+80=160 fits budget 200, but vram_mib 120+120=240 does not.
        s = Scheduler([p], budget_mib=200)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions())
            assert m1 not in s.resident  # evicted because of the buffer
            assert m2 in s.resident
            s.release(m2)
        finally:
            await s.stop()

    run(scenario())


# ---- Scheduler ----

def run(coro):
    return asyncio.run(coro)


def test_scheduler_routes_to_resident_and_release():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            m = await s.submit("m1", LoadOptions())
            assert m in s.resident
            assert s.in_flight[m] == 1
            s.release(m)
            assert s.in_flight[m] == 0
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_evicts_single_when_over_budget():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100, "m2": 80})
        s = Scheduler([p], budget_mib=150)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions())
            s.release(m1)
            m2 = await s.submit("m2", LoadOptions())
            assert m1 not in s.resident
            assert m2 in s.resident
            s.release(m2)
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_evicts_set_when_no_single():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80, "m3": 80, "m4": 100})
        s = Scheduler([p], budget_mib=240)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions()); s.release(m2)
            m3 = await s.submit("m3", LoadOptions()); s.release(m3)
            m4 = await s.submit("m4", LoadOptions())
            assert m1 not in s.resident
            assert m2 not in s.resident
            assert m3 in s.resident
            assert m4 in s.resident
            s.release(m4)
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_zero_memory_skips_eviction():
    async def scenario():
        p = MemProvider("http://a", {"a": 0, "b": 0})
        s = Scheduler([p], budget_mib=100)
        await s.start()
        try:
            ma = await s.submit("a", LoadOptions()); s.release(ma)
            mb = await s.submit("b", LoadOptions())
            assert ma in s.resident and mb in s.resident
            s.release(mb)
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_drains_in_flight_before_evict():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80, "m3": 80})
        s = Scheduler([p], budget_mib=100)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions())  # stays in-flight
            m3_task = asyncio.create_task(s.submit("m3", LoadOptions()))
            await asyncio.sleep(0)  # coordinator starts serving m3, blocks on drain
            assert m2 in s.resident  # not evicted yet
            s.release(m2)           # drain proceeds
            m3 = await m3_task
            assert m2 not in s.resident
            assert m3 in s.resident
            s.release(m3)
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_line_cut_serves_queued_eviction_request():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80, "m3": 80})
        s = Scheduler([p], budget_mib=100)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions()); s.release(m2)  # resident m2
            # enqueue A (m3) then B (m2); coordinator serves A, line-cuts B
            a_task = asyncio.create_task(s.submit("m3", LoadOptions()))
            b_task = asyncio.create_task(s.submit("m2", LoadOptions()))
            await asyncio.sleep(0)
            mb = await b_task  # B served (line-cut) -> same resident m2
            assert mb is m2
            s.release(mb)
            ma = await a_task  # now m2 quiescent -> evict m2, load m3
            assert ma.descriptor.modelId == "m3"
            s.release(ma)
            assert [x.descriptor.modelId for x in s.resident] == ["m3"]
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_model_not_found():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            with pytest.raises(ModelNotFound):
                await s.submit("nope", LoadOptions())
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_stop_flushes_pending_requests():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        # Enqueue a request; stop() must serve it while flushing.
        task = asyncio.create_task(s.submit("m1", LoadOptions()))
        stopping = asyncio.create_task(s.stop())
        await asyncio.sleep(0)  # coordinator serves the pending request
        model = await task  # served during stop
        assert model.descriptor.modelId == "m1"
        assert model in s.resident
        s.release(model)  # drain in-flight so stop() can finish
        await stopping

    run(scenario())


def test_scheduler_stop_waits_for_in_flight():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        m1 = await s.submit("m1", LoadOptions())  # stays in-flight
        stopping = asyncio.create_task(s.stop())
        await asyncio.sleep(0)
        assert not stopping.done()  # blocked on the in-flight request
        s.release(m1)  # drain proceeds
        await stopping

    run(scenario())


def test_scheduler_stop_times_out_when_in_flight_never_drains():
    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        m1 = await s.submit("m1", LoadOptions())  # never released
        # stop() must not hang forever: it force-tears-down after the drain
        # deadline even when an in-flight stream never completes.
        await s.stop(timeout=0.05)
        assert s._task is None

    run(scenario())


def test_scheduler_load_runs_off_thread():
    import threading

    async def scenario():
        p = MemProvider("http://a", {"m1": 100})
        seen = {}
        orig = p.loadModel

        def record(model):
            seen["thread"] = threading.get_ident()
            orig(model)

        p.loadModel = record
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            m = await s.submit("m1", LoadOptions())
            assert m in s.resident
            # loadModel ran on a worker thread, not the event-loop thread.
            assert seen["thread"] != threading.get_ident()
            s.release(m)
        finally:
            await s.stop()

    run(scenario())


# ---- on_start preload + protected eviction ----


def test_scheduler_protected_model_not_evicted():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80})
        s = Scheduler([p], budget_mib=150)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions())
            s.release(m1)
            s.protected.add("m1")
            # m2 would need to evict m1, but m1 is protected -> impossible.
            with pytest.raises(RuntimeError):
                await s.submit("m2", LoadOptions())
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_preload_once_model_evictable():
    async def scenario():
        p = MemProvider("http://a", {"a": 80, "b": 80})
        s = Scheduler([p], budget_mib=150)
        await s.start()
        try:
            await s.preload_on_start([("a", 4096, False)])
            assert [m.descriptor.modelId for m in s.resident] == ["a"]
            b = await s.submit("b", LoadOptions())
            s.release(b)
            # "once" models are preloaded but evictable like normal.
            assert [m.descriptor.modelId for m in s.resident] == ["b"]
            assert s.protected == set()
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_preload_always_protects_in_order():
    async def scenario():
        p = MemProvider("http://a", {"a": 60, "b": 60, "c": 60})
        s = Scheduler([p], budget_mib=200)
        await s.start()
        try:
            await s.preload_on_start(
                [("a", 4096, True), ("b", 4096, False), ("c", 4096, True)]
            )
            # Deterministic preload order, released (not in-flight), and the
            # "always" models are protected from eviction.
            assert [m.descriptor.modelId for m in s.resident] == ["a", "b", "c"]
            assert s.protected == {"a", "c"}
            assert all(s.in_flight[m] == 0 for m in s.resident)
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_preload_singular_oom_fails():
    async def scenario():
        p = MemProvider("http://a", {"big": 300})
        s = Scheduler([p], budget_mib=200)
        await s.start()
        try:
            # A singular model that cannot fit the budget fails startup.
            with pytest.raises(RuntimeError):
                await s.preload_on_start([("big", 4096, False)])
        finally:
            await s.stop()

    run(scenario())


def test_scheduler_impossible_load_raises():
    async def scenario():
        p = MemProvider("http://a", {"m1": 200})
        s = Scheduler([p], budget_mib=100)
        await s.start()
        try:
            # A runtime request that is impossible to load is refused with an
            # error and leaves no half-loaded resident model behind.
            with pytest.raises(RuntimeError):
                await s.submit("m1", LoadOptions())
            assert s.resident == []
        finally:
            await s.stop()

    run(scenario())


# ---- evict_idle (manual ctrl+e prune) ----


def test_evict_idle_evicts_idle_keeps_in_flight():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80, "m3": 80})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)  # idle
            m2 = await s.submit("m2", LoadOptions())  # stays in-flight
            m3 = await s.submit("m3", LoadOptions()); s.release(m3)  # idle
            await s.evict_idle()
            # m1/m3 were idle -> evicted; m2 still in-flight -> left resident.
            assert [m.descriptor.modelId for m in s.resident] == ["m2"]
            assert all(m._loaded is False for m in [m1, m3])
            s.release(m2)
        finally:
            await s.stop()

    run(scenario())


def test_evict_idle_skips_protected():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions()); s.release(m2)
            s.protected.add("m1")
            await s.evict_idle()
            # m1 protected stays resident; m2 idle non-protected is pruned.
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
        finally:
            await s.stop()

    run(scenario())


def test_evict_idle_noop_when_nothing_idle():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions())  # in-flight
            await s.evict_idle()
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
            s.release(m1)
        finally:
            await s.stop()

    run(scenario())


# ---- TTL auto-eviction ----


def test_ttl_prune_evicts_expired_keeps_fresh():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80, "m2": 80})
        s = Scheduler([p], budget_mib=1000, ttl=5)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            m2 = await s.submit("m2", LoadOptions()); s.release(m2)
            # m1 sat idle past the TTL; m2 finished only 1s ago -> stays.
            s.last_finish[m1] = time.monotonic() - 10
            s.last_finish[m2] = time.monotonic() - 1
            await s._prune(
                lambda m: time.monotonic() - s.last_finish.get(m, 0.0) >= s.ttl
            )
            assert [m.descriptor.modelId for m in s.resident] == ["m2"]
            assert m1._loaded is False
        finally:
            await s.stop()

    run(scenario())


def test_ttl_prune_skips_in_flight(monkeypatch):
    async def scenario():
        p = MemProvider("http://a", {"m1": 80})
        s = Scheduler([p], budget_mib=1000, ttl=5)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions())  # still in-flight
            s.last_finish[m1] = time.monotonic() - 10  # aged past TTL
            await s._prune(
                lambda m: time.monotonic() - s.last_finish.get(m, 0.0) >= s.ttl
            )
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
            s.release(m1)
        finally:
            await s.stop()

    run(scenario())


def test_ttl_loop_auto_evicts(monkeypatch):
    monkeypatch.setattr("scheduling.TTL_CHECK_INTERVAL", 0.05)

    async def scenario():
        p = MemProvider("http://a", {"m1": 80})
        s = Scheduler([p], budget_mib=1000, ttl=0.1)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            await asyncio.sleep(0.3)
            # The background loop should have auto-evicted m1 after >= ttl idle.
            assert m1 not in s.resident
        finally:
            await s.stop()

    run(scenario())


def test_ttl_disabled_no_eviction():
    async def scenario():
        p = MemProvider("http://a", {"m1": 80})
        s = Scheduler([p], budget_mib=1000, ttl=0)
        await s.start()
        try:
            m1 = await s.submit("m1", LoadOptions()); s.release(m1)
            s.last_finish[m1] = time.monotonic() - 100
            await asyncio.sleep(0.1)
            # ttl=0 disables the loop entirely: the idle model is kept.
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
        finally:
            await s.stop()

    run(scenario())


# ---- leases: claims that cannot outlive their request ----


class SlowLoadProvider(MemProvider):
    """MemProvider whose load blocks, so a request can be abandoned mid-load."""

    def __init__(self, endpoint_uri, memories, load_delay=0.0):
        super().__init__(endpoint_uri, memories)
        self._load_delay = load_delay
        self.load_calls = []
        self.unloads = []

    def loadModel(self, model):
        import time

        self.load_calls.append(model.descriptor.modelId)
        if self._load_delay:
            time.sleep(self._load_delay)
        model._loaded = True
        self._loaded.append(model)

    def unloadModel(self, model):
        model._loaded = False
        if model in self._loaded:
            self._loaded.remove(model)
        self.unloads.append(model.descriptor.modelId)


def test_lease_cancelled_during_load_leaves_no_claim():
    async def scenario():
        p = SlowLoadProvider("http://a", {"m1": 100}, load_delay=0.3)
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            lease = s.lease("m1", LoadOptions())
            task = asyncio.create_task(lease.acquire())
            await asyncio.sleep(0.05)  # parked in the load
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.4)  # the coordinator finishes the load
            assert p.load_calls == ["m1"]
            # The claim died with the requester: the model is ordinary idle.
            assert all(n == 0 for n in s.in_flight.values())
            await s.evict_idle()
            assert p.unloads == ["m1"]
        finally:
            await s.stop()

    run(scenario())


def test_lease_cancelled_while_queued_is_never_loaded():
    async def scenario():
        p = SlowLoadProvider("http://a", {"m1": 100, "m2": 100}, load_delay=0.3)
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            first = s.lease("m1", LoadOptions())
            second = s.lease("m2", LoadOptions())
            t1 = asyncio.create_task(first.acquire())
            t2 = asyncio.create_task(second.acquire())
            await asyncio.sleep(0.05)  # m1 loading, m2 queued behind it
            second.release()
            t2.cancel()
            await asyncio.gather(t2, return_exceptions=True)
            await t1
            assert p.load_calls == ["m1"], "abandoned request must not load"
            assert [m.descriptor.modelId for m in s.resident] == ["m1"]
            first.release()
            assert all(n == 0 for n in s.in_flight.values())
            await s.evict_idle()
            assert p.unloads == ["m1"]
        finally:
            await s.stop()

    run(scenario())


def test_lease_release_is_idempotent():
    async def scenario():
        p = SlowLoadProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            lease = s.lease("m1", LoadOptions())
            await lease.acquire()
            lease.release()
            lease.release()
            lease.release()
            assert all(n == 0 for n in s.in_flight.values())
        finally:
            await s.stop()

    run(scenario())


def test_lease_release_before_acquire_owns_nothing():
    async def scenario():
        p = SlowLoadProvider("http://a", {"m1": 100})
        s = Scheduler([p], budget_mib=1000)
        await s.start()
        try:
            lease = s.lease("m1", LoadOptions())
            lease.release()
            with pytest.raises(asyncio.CancelledError):
                await lease.acquire()
            await asyncio.sleep(0.1)
            assert p.load_calls == []
            assert all(n == 0 for n in s.in_flight.values())
        finally:
            await s.stop()

    run(scenario())
