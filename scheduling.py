import asyncio
from collections import defaultdict

from abstractions.model import Model
from abstractions.provider import Provider
from abstractions.routing import lookup_model


class ModelNotFound(Exception):
    def __init__(self, model_id: str) -> None:
        super().__init__(f"model not found: {model_id}")
        self.model_id = model_id


def select_evictions(resident: list[Model], shortfall_mib: float) -> list[Model]:
    """Pick the least-impact set of resident models to evict to free shortfall.

    Candidate A: the smallest single model that alone frees >= shortfall.
    Candidate B: greedily accumulate smallest-to-next-smallest until >= shortfall.
    Return whichever set is closer to shortfall (smaller over-eviction),
    tie-breaking toward the single model. Raise if neither can free enough.
    """
    candidates = [m for m in resident if m.memory() > 0]

    a = min(
        (m for m in candidates if m.memory() >= shortfall_mib),
        key=lambda m: m.memory(),
        default=None,
    )

    b = []
    total = 0.0
    for m in sorted(candidates, key=lambda m: m.memory()):
        b.append(m)
        total += m.memory()
        if total >= shortfall_mib:
            break
    if total < shortfall_mib:
        b = None

    if a is None and b is None:
        raise RuntimeError("cannot free enough VRAM to fit model")
    if a is None:
        return b
    if b is None:
        return [a]

    a_freed = a.memory()
    b_freed = total
    if a_freed - shortfall_mib <= b_freed - shortfall_mib:
        return [a]
    return b


class Scheduler:
    def __init__(self, providers: list[Provider], budget_mib: float) -> None:
        self.providers = providers
        self.budget_mib = budget_mib
        self.resident: list[Model] = []
        self.pending: list[tuple] = []  # (model_id, load_options, future)
        self.in_flight: dict[Model, int] = defaultdict(int)
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._wake.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def current_free(self) -> float:
        used = sum(m.memory() for m in self.resident)
        return self.budget_mib - used

    def _resident_for(self, provider: Provider, model_id: str):
        for m in self.resident:
            if m.descriptor.modelId == model_id:
                return m
        if getattr(provider, "single_resident", False):
            for m in self.resident:
                if m.descriptor.provider is provider:
                    return m
        return None

    def _descriptor_for(self, provider: Provider, model_id: str):
        for d in provider.getModelsDescriptors():
            if d.modelId == model_id:
                return d
        return None

    async def submit(self, model_id: str, load_options) -> Model:
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((model_id, load_options, fut))
        self._wake.set()
        return await fut

    def release(self, model: Model) -> None:
        if self.in_flight[model] > 0:
            self.in_flight[model] -= 1

    async def _run(self) -> None:
        while True:
            while self.pending:
                model_id, load_options, fut = self.pending.pop(0)
                try:
                    result = await self._serve(model_id, load_options)
                    if not fut.done():
                        fut.set_result(result)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
            await self._wake.wait()
            self._wake.clear()

    async def _serve(self, model_id: str, load_options) -> Model:
        provider = lookup_model(self.providers, model_id)
        if provider is None:
            raise ModelNotFound(model_id)
        resident = self._resident_for(provider, model_id)
        if resident is not None:
            self.in_flight[resident] += 1
            return resident

        descriptor = self._descriptor_for(provider, model_id)
        model = provider.createModel(descriptor, load_options)
        mem = model.memory()
        if mem > 0:
            shortfall = mem - self.current_free()
            if shortfall > 0:
                to_evict = select_evictions(self.resident, shortfall)
                await self._quiesce(to_evict)
                for m in to_evict:
                    m.descriptor.provider.unloadModel(m)
                self.resident = [m for m in self.resident if m not in to_evict]
        provider.loadModel(model)
        self.resident.append(model)
        self.in_flight[model] += 1
        return model

    def _targets(self, model_id: str, eviction_models: list[Model]) -> bool:
        for m in eviction_models:
            if self._resident_for(m.descriptor.provider, model_id) is m:
                return True
        return False

    async def _quiesce(self, eviction_models: list[Model]) -> None:
        # Line-cutting: serve queued requests for eviction models first.
        while True:
            idx = None
            for i, (mid, lo, fut) in enumerate(self.pending):
                if self._targets(mid, eviction_models):
                    idx = i
                    break
            if idx is not None:
                mid, lo, fut = self.pending.pop(idx)
                try:
                    result = await self._serve(mid, lo)
                    if not fut.done():
                        fut.set_result(result)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            # Drain: wait until no eviction model has outstanding requests.
            if not any(self.in_flight[m] > 0 for m in eviction_models):
                return
            await asyncio.sleep(0)
