import asyncio

import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from scheduling import ModelNotFound, Scheduler, select_evictions


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
