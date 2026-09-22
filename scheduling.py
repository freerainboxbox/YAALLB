import asyncio
import time
from collections import defaultdict

import log
from abstractions.load_options import LoadOptions
from abstractions.model import Model
from abstractions.provider import Provider
from abstractions.routing import lookup_model


def _provider_label(provider: Provider) -> str:
    return f"{provider._type_id}#{getattr(provider, '_instance_id', 0)}"


def _impact_suffix(prev_used: float, impact: float, is_evict: bool) -> str:
    """Append the VRAM impact of a load/eviction to a log line.

    Formats `` <sign><impact> MiB computed impact (<prev> -> <new>)``, where
    sign is '+' for a load and '-' for an eviction, and ``new = prev +/- 
    impact`` — e.g. 1000 MiB already loaded and a 2000 MiB model incoming
    yields `` +2000 MiB computed impact (1000 -> 3000)``.
    """
    delta = -impact if is_evict else impact
    new_used = prev_used + delta
    return f" {delta:+.0f} MiB computed impact ({prev_used:.0f} -> {new_used:.0f})"


# How long stop() waits for queued/in-flight requests to drain before force
# tearing down the coordinator, so a stuck upstream or disconnected client
# can't hang graceful shutdown forever.
STOP_DRAIN_TIMEOUT = 30.0

# Poll interval while waiting for in-flight requests to drain (stop() and
# eviction quiesce). sleep(0) would busy-spin a core for the whole drain,
# which lasts as long as the longest in-flight generation.
DRAIN_POLL_INTERVAL = 0.05

# How often the TTL auto-eviction task scans resident models for idle ones
# that have sat unserved for >= ttl seconds. A 1s scan keeps eviction prompt
# without much overhead.
TTL_CHECK_INTERVAL = 1.0

# How long reallocation waits on an eviction target's in-flight I/O before it
# starts saying so out loud. Draining is correct (a running generation is
# never cut off) but an endless wait freezes the single coordinator, so a long
# stall has to be visible in the log rather than looking like a hang.
QUIESCE_STALL_NOTICE = 30.0


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
    candidates = [m for m in resident if m.vram_mib() > 0]

    a = min(
        (m for m in candidates if m.vram_mib() >= shortfall_mib),
        key=lambda m: m.vram_mib(),
        default=None,
    )

    b = []
    total = 0.0
    for m in sorted(candidates, key=lambda m: m.vram_mib()):
        b.append(m)
        total += m.vram_mib()
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

    a_freed = a.vram_mib()
    b_freed = total
    if a_freed - shortfall_mib <= b_freed - shortfall_mib:
        return [a]
    return b


class Lease:
    """One in-flight claim on one model, held by exactly one request.

    Claims are what keep a model out of every eviction path (the ctrl+e prune,
    the TTL sweep and budget reallocation all skip in-flight models), so a
    claim that outlives its request makes that model unevictable until the
    process restarts. Requests die at arbitrary await points -- a client that
    hangs up mid-load is cancelled out from under the route, with no chance to
    run cleanup that lives on a happy path -- so the claim is owned by this
    object instead of by the route's code: `release()` is synchronous and
    idempotent, and is correct before the grant, during a load, or after it.
    """

    def __init__(
        self, scheduler: "Scheduler", model_id: str, load_options
    ) -> None:
        self._scheduler = scheduler
        self.model_id = model_id
        self.load_options = load_options
        self.model: Model | None = None
        self.future = asyncio.get_running_loop().create_future()
        self._granted = False
        # True once the requester is done with the lease (released or
        # abandoned): it owns nothing, and the coordinator must not hand it a
        # model it will never give back.
        self._finished = False

    async def acquire(self) -> Model:
        """Queue for the model (loading it if needed) and hold a claim on it.

        Cancellable at every await: giving up here returns the claim, or takes
        the queued slot back, so nothing is left pointing at a requester that
        is no longer there.
        """
        if self._finished:
            raise asyncio.CancelledError("lease already released")
        self._scheduler.pending.append(self)
        self._scheduler._wake.set()
        try:
            return await self.future
        except BaseException:
            # Covers the client-disconnect cancellation Starlette delivers as
            # well as a scheduler-side failure: either way this requester will
            # never hand the model back, so give it up right here.
            self.release()
            raise

    def release(self) -> None:
        """Give up the claim, or the queued slot. Idempotent and synchronous.

        Synchronous on purpose: the caller calls it first thing in a `finally`
        while unwinding a cancelled task, where any await can be re-cancelled
        and would skip whatever came after it.
        """
        if self._finished:
            return
        self._finished = True
        if self._granted:
            self._scheduler._leave(self.model)
            return
        # Never granted: drop the queued slot so the coordinator does not load
        # a model for a client that is already gone. If the coordinator is
        # mid-serve on this lease right now, `_finished` makes it take the
        # claim back the moment it grants it.
        if self in self._scheduler.pending:
            self._scheduler.pending.remove(self)
        if not self.future.done():
            self.future.cancel()


class Scheduler:
    def __init__(
        self, providers: list[Provider], budget_mib: float, ttl: float | None = None
    ) -> None:
        self.providers = providers
        self.budget_mib = budget_mib
        # TTL in seconds: an idle model that has not served a request for >= ttl
        # is auto-evicted. None/0/negative disables the feature.
        self.ttl = ttl or 0.0
        self.resident: list[Model] = []
        self.pending: list[Lease] = []  # queued requests, waiting to be served
        self.in_flight: dict[Model, int] = defaultdict(int)
        # Requests the coordinator has dequeued but not granted yet (i.e. a
        # load in progress). Keeps stop() from calling the system quiescent
        # while a model is half-loaded.
        self.serving = 0
        # monotonic() timestamp of when each model last finished a request.
        self.last_finish: dict[Model, float] = {}
        # Model ids that must never be evicted (on_start "always" models).
        self.protected: set[str] = set()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._ttl_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            self._ttl_task = asyncio.create_task(self._ttl_loop())

    async def stop(self, timeout: float = STOP_DRAIN_TIMEOUT) -> None:
        if self._task is None:
            return
        # Graceful shutdown: flush queued requests and wait for in-flight ones
        # to finish before tearing down the coordinator. The drain is bounded
        # by a deadline so a stuck upstream (or disconnected client) that never
        # releases its model can't hang shutdown indefinitely.
        deadline = asyncio.get_running_loop().time() + timeout
        while self.pending or self.serving or any(self.in_flight.values()):
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(DRAIN_POLL_INTERVAL)
        self._wake.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        # The coordinator is gone, so nothing left will ever serve what is
        # still queued. Hand those leases back so their requests fail fast
        # instead of waiting on a future that can no longer be resolved.
        for lease in list(self.pending):
            lease.release()
        if self._ttl_task is not None:
            self._ttl_task.cancel()
            try:
                await self._ttl_task
            except asyncio.CancelledError:
                pass
            self._ttl_task = None
        self._task = None

    def current_free(self) -> float:
        used = sum(m.vram_mib() for m in self.resident)
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

    def lease(self, model_id: str, load_options) -> Lease:
        """Create a cancellation-safe claim on `model_id`.

        The request path owns the returned lease: `await lease.acquire()`, then
        `lease.release()` in a `finally`. Prefer this over `submit()`.
        """
        return Lease(self, model_id, load_options)

    async def submit(self, model_id: str, load_options) -> Model:
        """Queue for a model and hand back a raw in-flight claim on it.

        The caller owes exactly one `release(model)` for the returned model.
        That handoff has no protection against a caller that is cancelled
        before it reaches its release, which is how an aborted client used to
        strand a model in-flight forever; new code takes `lease()` instead.
        """
        return await self.lease(model_id, load_options).acquire()

    def release(self, model: Model) -> None:
        """Give back a raw claim taken by `submit()`. See `lease()`."""
        self._leave(model)

    def _leave(self, model: Model) -> None:
        if self.in_flight[model] > 0:
            self.in_flight[model] -= 1
            if self.in_flight[model] == 0:
                self.last_finish[model] = time.monotonic()

    def _grant(self, lease: Lease, model: Model) -> None:
        """Hand a ready model to its requester, or take the claim straight back.

        A requester that gave up while the model was loading still gets here:
        the model stays resident as an ordinary idle model (so ctrl+e and the
        TTL sweep can evict it) instead of holding a claim nobody owns.
        """
        self.in_flight[model] += 1
        lease.model = model
        lease._granted = True
        if lease._finished or lease.future.cancelled():
            self._leave(model)
            return
        if not lease.future.done():
            lease.future.set_result(model)

    def _reject(self, lease: Lease, exc: Exception) -> None:
        if lease._finished or lease.future.done():
            return
        lease.future.set_exception(exc)

    async def _prune(self, predicate) -> None:
        """Unload every resident model that is idle, non-protected, and matches
        predicate.

        In-flight models are left resident — their eviction is *not* queued —
        so a running generation is never cut off. Used by the manual ctrl+e
        prune (predicate always True) and the TTL auto-eviction.
        """
        to_evict = [
            m for m in self.resident
            if self.in_flight[m] == 0
            and m.descriptor.modelId not in self.protected
            and predicate(m)
        ]
        if not to_evict:
            return
        running = sum(m.vram_mib() for m in self.resident)
        for m in to_evict:
            impact = m.vram_mib()
            running -= impact
            log.warning(
                f"prune model={m.descriptor.modelId} "
                f"provider={_provider_label(m.descriptor.provider)}"
                + _impact_suffix(running + impact, impact, is_evict=True)
            )
            await asyncio.to_thread(m.descriptor.provider.unloadModel, m)
        self.resident = [m for m in self.resident if m not in to_evict]

    async def evict_idle(self) -> None:
        """Prune every resident model that is not actively serving a request.

        This is the cleanup path for a manual ctrl+e; it never cuts off a
        running generation and skips protected models.
        """
        await self._prune(lambda m: True)

    async def _ttl_loop(self) -> None:
        """Periodically auto-evict models idle for >= ttl seconds."""
        if not self.ttl or self.ttl <= 0:
            return
        while True:
            await asyncio.sleep(TTL_CHECK_INTERVAL)
            now = time.monotonic()
            await self._prune(
                lambda m: now - self.last_finish.get(m, 0.0) >= self.ttl
            )

    async def _run(self) -> None:
        while True:
            while self.pending:
                await self._serve(self.pending.pop(0))
            await self._wake.wait()
            self._wake.clear()

    async def _serve(self, lease: Lease) -> None:
        """Serve one dequeued request: resolve, evict if needed, load, grant.

        Results travel back through the lease's future, and every exit path
        either grants the model or leaves the request holding nothing.
        """
        model_id, load_options = lease.model_id, lease.load_options
        if lease._finished:
            # The client hung up before the coordinator got to this request:
            # do not spend a load (and a claim) on nobody.
            return
        # A load in progress is neither pending nor in-flight yet; count it so
        # stop() won't see a served-but-unloaded request as quiescent and tear
        # down the coordinator early.
        self.serving += 1
        try:
            # Descriptor lookups can block on provider HTTP (LM Studio TTL
            # miss), so run them off the event loop like memory()/loadModel
            # below; otherwise one slow provider stalls every in-flight relay.
            provider = await asyncio.to_thread(
                lookup_model, self.providers, model_id
            )
            if provider is None:
                self._reject(lease, ModelNotFound(model_id))
                return
            resident = self._resident_for(provider, model_id)
            if resident is not None:
                self._grant(lease, resident)
                return

            descriptor = await asyncio.to_thread(
                self._descriptor_for, provider, model_id
            )
            model = provider.createModel(descriptor, load_options)
            mem = await asyncio.to_thread(model.vram_mib)
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
                    running = sum(m.vram_mib() for m in self.resident)
                    for m in to_evict:
                        impact = m.vram_mib()
                        running -= impact
                        log.warning(
                            f"unload model={m.descriptor.modelId} "
                            f"provider={_provider_label(m.descriptor.provider)}"
                            + _impact_suffix(running + impact, impact, is_evict=True)
                        )
                        m.descriptor.provider.unloadModel(m)
                    self.resident = [m for m in self.resident if m not in to_evict]
            used_before_load = sum(m.vram_mib() for m in self.resident)
            log.warning(
                f"load model={model_id} "
                f"provider={_provider_label(provider)} "
                f"ctx={load_options.ctx_length}"
                + _impact_suffix(used_before_load, mem, is_evict=False)
            )
            await asyncio.to_thread(provider.loadModel, model)
            self.resident.append(model)
            self._grant(lease, model)
        except Exception as e:
            self._reject(lease, e)
        finally:
            self.serving -= 1

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
            lease = self.lease(model_id, LoadOptions(ctx_length=ctx_length))
            try:
                await lease.acquire()
            finally:
                lease.release()

    def _targets(self, model_id: str, eviction_models: list[Model]) -> bool:
        for m in eviction_models:
            if self._resident_for(m.descriptor.provider, model_id) is m:
                return True
        return False

    async def _quiesce(self, eviction_models: list[Model]) -> None:
        # Line-cutting: serve queued requests for eviction models first.
        loop = asyncio.get_running_loop()
        started = loop.time()
        last_notice = 0.0
        while True:
            idx = None
            for i, lease in enumerate(self.pending):
                if self._targets(lease.model_id, eviction_models):
                    idx = i
                    break
            if idx is not None:
                await self._serve(self.pending.pop(idx))
                continue
            # Drain: wait until no eviction model has outstanding requests.
            if not any(self.in_flight[m] > 0 for m in eviction_models):
                return
            waited = loop.time() - started
            if waited - last_notice >= QUIESCE_STALL_NOTICE:
                last_notice = waited
                log.warning(
                    "reallocate: waiting "
                    f"{int(waited)}s for in-flight "
                    f"[{', '.join(m.descriptor.modelId for m in eviction_models)}] "
                    "to drain"
                )
            await asyncio.sleep(DRAIN_POLL_INTERVAL)
