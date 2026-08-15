import asyncio
from collections import defaultdict

import log
from abstractions.load_options import LoadOptions
from abstractions.model import Model
from abstractions.provider import Provider
from abstractions.routing import lookup_model


def _provider_label(provider: Provider) -> str:
    return f"{provider._type_id}#{getattr(provider, '_instance_id', 0)}"


# How long stop() waits for queued/in-flight requests to drain before force
# tearing down the coordinator, so a stuck upstream or disconnected client
# can't hang graceful shutdown forever.
STOP_DRAIN_TIMEOUT = 30.0

# Poll interval while waiting for in-flight requests to drain (stop() and
# eviction quiesce). sleep(0) would busy-spin a core for the whole drain,
# which lasts as long as the longest in-flight generation.
DRAIN_POLL_INTERVAL = 0.05


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
        # Model ids that must never be evicted (on_start "always" models).
        self.protected: set[str] = set()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self, timeout: float = STOP_DRAIN_TIMEOUT) -> None:
        if self._task is None:
            return
        # Graceful shutdown: flush queued requests and wait for in-flight ones
        # to finish before tearing down the coordinator. The drain is bounded
        # by a deadline so a stuck upstream (or disconnected client) that never
        # releases its model can't hang shutdown indefinitely.
        deadline = asyncio.get_running_loop().time() + timeout
        while self.pending or any(self.in_flight.values()):
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(DRAIN_POLL_INTERVAL)
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
        # Descriptor lookups can block on provider HTTP (LM Studio TTL
        # miss), so run them off the event loop like memory()/loadModel
        # below; otherwise one slow provider stalls every in-flight relay.
        provider = await asyncio.to_thread(lookup_model, self.providers, model_id)
        if provider is None:
            raise ModelNotFound(model_id)
        resident = self._resident_for(provider, model_id)
        if resident is not None:
            self.in_flight[resident] += 1
            return resident

        descriptor = await asyncio.to_thread(self._descriptor_for, provider, model_id)
        model = provider.createModel(descriptor, load_options)
        # Mark in-flight before the slow load so stop() won't see a served-but-
        # unloaded request as quiescent and tear down the coordinator early.
        self.in_flight[model] += 1
        try:
            mem = await asyncio.to_thread(model.memory)
            if mem > 0:
                shortfall = mem - self.current_free()
                if shortfall > 0:
                    # Protected (on_start "always") models are never evicted;
                    # exclude them so select_evictions raises if they alone
                    # can't free enough VRAM.
                    evictable = [
                        m for m in self.resident
                        if m.descriptor.modelId not in self.protected
                    ]
                    to_evict = select_evictions(evictable, shortfall)
                    log.warning(
                        f"reallocate: new model={model_id} "
                        f"provider={_provider_label(provider)} "
                        f"evicting=[{', '.join(m.descriptor.modelId for m in to_evict)}]"
                    )
                    await self._quiesce(to_evict)
                    for m in to_evict:
                        log.warning(
                            f"unload model={m.descriptor.modelId} "
                            f"provider={_provider_label(m.descriptor.provider)}"
                        )
                        m.descriptor.provider.unloadModel(m)
                    self.resident = [m for m in self.resident if m not in to_evict]
            log.warning(
                f"load model={model_id} "
                f"provider={_provider_label(provider)} "
                f"ctx={load_options.ctx_length}"
            )
            await asyncio.to_thread(provider.loadModel, model)
            self.resident.append(model)
            return model
        except Exception:
            self.in_flight[model] -= 1
            raise

    async def preload_on_start(self, targets: list[tuple]) -> None:
        """Preload on_start models in deterministic order.

        targets is a list of (model_id, ctx_length, protected). protected
        ('always') models are marked non-evictable before loading; 'once'
        models are evicted like normal. A model that cannot fit the budget
        even alone raises RuntimeError, which fails startup.
        """
        for model_id, ctx_length, protect in targets:
            if protect:
                self.protected.add(model_id)
            try:
                model = await self.submit(
                    model_id, LoadOptions(ctx_length=ctx_length)
                )
                self.release(model)
            except Exception:
                raise

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
            await asyncio.sleep(DRAIN_POLL_INTERVAL)
