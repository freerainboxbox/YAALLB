import os.path
import subprocess

import log
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from abstractions.ready import wait_server_ready
from providers.dwarfstar_estimate import (
    Ds4BuildError,
    build_estimator,
    estimate as ds4_estimate,
    estimate_mib as ds4_estimate_mib,
)

# ds4 cannot answer a native /v1/models while it is spawned/terminated by
# Python, so its model list is built here. Both model IDs point to the same
# underlying model; the presented context_length is 1000000 (DeepSeek v4's
# maximum) unless a model is resident with a different ctx_length.
DS4_CONTEXT_LENGTH = 1000000

DS4_DEFAULT_HOST = "127.0.0.1"
DS4_DEFAULT_PORT = 8000
DS4_DEFAULT_BINARY = "./ds4-server"

# YAALLB asks the ds4 build itself what a serve configuration will occupy (see
# providers/dwarfstar_estimate.py and tools/ds4_estimate.c). The estimator is a
# second binary next to ds4-server, built from the same tree by
# tools/ds4-estimate.mk, and is resolved relative to ds4_dir just like the
# server binary.
DS4_DEFAULT_ESTIMATOR = "./ds4-estimate"

# How long loadModel waits for the spawned ds4-server to start accepting
# requests before failing the load. ds4 loads the model at launch, so a 200
# from /v1/models means the model is resident and ready.
DS4_READY_TIMEOUT = 120

# Flag registry: config key -> (flag, kind, default). Defaults grabbed from
# `./ds4-server --help`. `--ctx` is deliberately absent: it comes from
# LoadOptions.ctx_length at load time.
#
# Also accepted by ds4-server but deliberately not exposed here: `--chdir` (the
# provider already sets the working directory), and the distributed /
# tensor-parallel / `--dir-steering-*` families, which move or re-slice the
# model in ways YAALLB's footprint projection does not describe.
DS4_OPTIONS = {
    "vision": ("--vision", "value", None),
    "backend": ("--backend", "value", None),
    "metal": ("--metal", "flag", False),
    "cuda": ("--cuda", "flag", False),
    "rocm": ("--rocm", "flag", False),
    "cpu": ("--cpu", "flag", False),
    "gpu_vram": ("--gpu-vram", "value", None),
    "gpu_devices": ("--gpu-devices", "value", None),
    "cuda_tensor_parallel": ("--cuda-tensor-parallel", "flag", False),
    "tokens": ("-n", "value", None),
    "threads": ("-t", "value", None),
    "power": ("--power", "value", 100),
    "ssd_streaming": ("--ssd-streaming", "flag", False),
    "ssd_streaming_cold": ("--ssd-streaming-cold", "flag", False),
    "ssd_streaming_cache_experts": ("--ssd-streaming-cache-experts", "value", None),
    "ssd_streaming_full_layers": ("--ssd-streaming-full-layers", "value", None),
    "ssd_streaming_preload_experts": ("--ssd-streaming-preload-experts", "value", None),
    "simulate_used_memory": ("--simulate-used-memory", "value", None),
    "prefill_chunk": ("--prefill-chunk", "value", None),
    "mtp": ("--mtp", "flag", False),
    "mtp_model": ("--mtp-model", "value", None),
    "mtp_draft": ("--mtp-draft", "value", 1),
    "mtp_margin": ("--mtp-margin", "value", 3),
    "mtp_timing": ("--mtp-timing", "flag", False),
    "dspark": ("--dspark", "flag", False),
    # ds4 picks the confidence threshold per backend and sampling mode (Metal
    # 0.6, CUDA/ROCm 0.7, exact sampling 0.8), so there is no single default
    # to withhold: any configured value is passed.
    "dspark_confidence": ("--dspark-confidence", "value", None),
    "mtp_exact_sampling": ("--mtp-exact-sampling", "flag", False),
    "dspark_strict": ("--dspark-strict", "flag", False),
    "quality": ("--quality", "flag", False),
    "warm_weights": ("--warm-weights", "flag", False),
    "cors": ("--cors", "flag", False),
    "trace": ("--trace", "value", None),
    "batched_session": ("--batched-session", "value", None),
    "mixed_prefill_quantum": ("--mixed-prefill-quantum", "value", 128),
    "kv_disk_dir": ("--kv-disk-dir", "value", None),
    "kv_disk_space_mb": ("--kv-disk-space-mb", "value", 4096),
    "kv_cache_min_tokens": ("--kv-cache-min-tokens", "value", 512),
    "kv_cache_cold_max_tokens": ("--kv-cache-cold-max-tokens", "value", 30000),
    "kv_cache_continued_interval_tokens": (
        "--kv-cache-continued-interval-tokens",
        "value",
        10000,
    ),
    "kv_cache_boundary_trim_tokens": ("--kv-cache-boundary-trim-tokens", "value", 32),
    "kv_cache_boundary_align_tokens": ("--kv-cache-boundary-align-tokens", "value", 2048),
    "kv_cache_reject_different_quant": ("--kv-cache-reject-different-quant", "flag", False),
    "disable_exact_dsml_tool_replay": ("--disable-exact-dsml-tool-replay", "flag", False),
    "tool_memory_max_ids": ("--tool-memory-max-ids", "value", 100000),
}


class DwarfStarProvider(Provider):
    _type_id = "ds4"
    single_resident = True

    class Model(BaseModel):
        def memory(self) -> float:
            # Footprint components come from the ds4 build itself (see
            # providers/dwarfstar_estimate.py), because they depend on the
            # shape in the GGUF, the backend, the effective prefill chunk, the
            # SSD streaming mode, and any drafter/MTP support GGUF.
            provider = self.descriptor.provider
            ctx = provider._effective_ctx(self)
            return ds4_estimate_mib(provider._estimate(ctx), provider._sessions())

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = DS4_DEFAULT_HOST
        self.port = DS4_DEFAULT_PORT
        self.ds4_dir: str | None = None
        self.gguf_path: str | None = None
        self.binary: str = DS4_DEFAULT_BINARY
        self.estimate_binary: str = DS4_DEFAULT_ESTIMATOR
        self.options: dict = {}
        self.ctx_length: int | None = None
        self.resident_model: BaseModel | None = None
        self._process: subprocess.Popen | None = None
        super().__init__(_instance_id, config)

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _effective_ctx(self, model: BaseModel | None = None) -> int:
        # ds4 sets --ctx once at startup and both served models inherit it, so
        # the provider-level ctx_length (when set) overrides any per-model one.
        # A loading model is passed explicitly: at spawn time resident_model is
        # still None, so without it the fallback DS4_CONTEXT_LENGTH would be
        # spawned while memory()//v1/models account the request's ctx.
        if self.ctx_length is not None:
            return self.ctx_length
        if model is not None:
            return model.loadOptions.ctx_length
        if self.resident_model is not None:
            return self.resident_model.loadOptions.ctx_length
        return DS4_CONTEXT_LENGTH

    def _backend_name(self) -> str | None:
        """ds4 backend name for the configured flags.

        None leaves ds4's own platform default in charge, so the estimator and
        the server agree without YAALLB restating the default.
        """
        backend = self.options.get("backend")
        if backend:
            return str(backend)
        for name in ("metal", "cuda", "rocm", "cpu"):
            if self.options.get(name):
                return name
        return None

    def _sessions(self) -> int:
        # ds4-server keeps one session, or `--batched-session N` resident
        # sessions, and gives each its own session graphs/caches and its own
        # drafter scratch, so both per-session terms are multiplied.
        try:
            sessions = int(self.options.get("batched_session") or 1)
        except (TypeError, ValueError):
            return 1
        return max(sessions, 1)

    def _estimate(self, ctx: int) -> dict:
        """Footprint components (bytes) for one ctx, memoized by the provider."""
        return ds4_estimate(
            ds4_dir=self.ds4_dir or ".",
            gguf_path=self.gguf_path or "",
            ctx=ctx,
            binary=self.estimate_binary or DS4_DEFAULT_ESTIMATOR,
            backend=self._backend_name(),
            prefill_chunk=self.options.get("prefill_chunk"),
            ssd_streaming=bool(self.options.get("ssd_streaming", False)),
            mtp_model=self.options.get("mtp_model"),
            vision=self.options.get("vision"),
            # --dspark decides whether a session also gets the verifier graph;
            # the DSpark capture buffers and the support GGUF are budgeted
            # whether or not it is on.
            dspark=bool(self.options.get("dspark", False)),
        )

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor("deepseek-v4-flash", self),
            ModelDescriptor("deepseek-v4-pro", self),
        ]

    def getOAIModels(self) -> list[dict]:
        ctx_length = self._effective_ctx()

        def model_entry(model_id: str) -> dict:
            return {
                "id": model_id,
                "object": "model",
                "created": 1767225600,
                "owned_by": "ds4.c",
                "name": "DeepSeek V4 Flash",
                "context_length": ctx_length,
                "top_provider": {
                    "context_length": DS4_CONTEXT_LENGTH,
                    "max_completion_tokens": 393216,
                    "is_moderated": False,
                },
                "supported_parameters": [
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "stop",
                    "seed",
                    "stream",
                    "reasoning_effort",
                ],
            }

        return [model_entry("deepseek-v4-flash"), model_entry("deepseek-v4-pro")]

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def _build_command(self, model: BaseModel) -> list[str]:
        ctx_length = self._effective_ctx(model)

        command = [self.binary, "-m", self.gguf_path]

        if self.host != DS4_DEFAULT_HOST:
            command += ["--host", self.host]
        if self.port != DS4_DEFAULT_PORT:
            command += ["--port", str(self.port)]

        for key, (flag, kind, default) in DS4_OPTIONS.items():
            if key not in self.options:
                continue
            value = self.options[key]
            if kind == "flag":
                if value is True:
                    command.append(flag)
            elif value != default:
                command += [flag, str(value)]

        command += ["--ctx", str(ctx_length)]
        return command

    def loadModel(self, model: BaseModel) -> None:
        if self.ds4_dir is None:
            raise ValueError("ds4_dir must be set in config before loading")
        if self.gguf_path is None:
            raise ValueError("gguf_path must be set in config before loading")

        command = self._build_command(model)
        self._process = subprocess.Popen(command, cwd=self.ds4_dir)
        self.resident_model = model
        # The model is loading until the spawned server actually accepts
        # requests; only then is it marked loaded (ready).
        model._loaded = False
        model._load_state = "loading"
        try:
            wait_server_ready(
                self.endpoint_uri, self._process, "ds4-server", DS4_READY_TIMEOUT
            )
        except Exception:
            # A readiness timeout (or server exit) must not orphan the spawned
            # ds4-server: terminate it and clear provider state before the
            # load fails, so a VRAM-holding child isn't leaked.
            self.unloadModel(model)
            raise
        model._loaded = True
        model._load_state = "ready"

    def unloadModel(self, model: BaseModel) -> None:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # wait(timeout) raises instead of returning, so a ds4 that
                # ignores SIGTERM (stuck in a Metal kernel or KV disk I/O)
                # must be escalated to SIGKILL here; letting the exception
                # escape would wedge the scheduler's eviction path.
                process.kill()
                process.wait()
            self._process = None
        self.resident_model = None
        model._loaded = False


def build_dwarfstar_estimators(providers: list) -> None:
    """Build the footprint estimator into every ds4 tree config.json names.

    YAALLB asks the ds4 build itself what a serve configuration will occupy (see
    providers/dwarfstar_estimate.py), and the answer is only valid for the exact
    build it was linked against, so the estimator belongs inside each tree. It
    is YAALLB's own program: building it *adds* two files next to ds4-server
    (`ds4-estimate`, `ds4_estimate.host.o`) and rewrites nothing, and make
    rebuilds nothing when the tree is already current - which is also why the
    estimator never goes stale against an updated ds4 again.

    Several instances of one tree share one build. Anything that would leave a
    ds4 instance unbudgeted raises: startup must not continue on a guess.
    """
    roots: list[str] = []
    for provider in providers:
        if provider._type_id != DwarfStarProvider._type_id:
            continue
        ds4_dir = getattr(provider, "ds4_dir", None)
        if not ds4_dir:
            raise Ds4BuildError(
                f"ds4 provider #{provider._instance_id} has no ds4_dir, so its "
                "footprint estimator cannot be built and its models cannot be "
                "budgeted"
            )
        root = os.path.abspath(ds4_dir)
        if root not in roots:
            roots.append(root)

    for root in roots:
        build_estimator(root)
        log.info(
            f"ds4 estimator ready dir={root} "
            "(added to the tree: ds4-estimate, ds4_estimate.host.o)"
        )


def warm_dwarfstar_estimates(providers: list, default_ctx: int | None = None) -> None:
    """Precompute each ds4 provider's footprint for the contexts it will serve.

    `Model.memory()` runs in the request path, while one estimator run costs a
    subprocess plus a GGUF metadata read (~0.3s measured; no weights are
    loaded). Warming the configured context lengths at startup keeps the first
    request off that cost; requests that ask for a different ctx still get an
    estimate, they just pay one estimator run for it (see
    providers/dwarfstar_estimate.py).
    """
    ds4_providers = [p for p in providers if p._type_id == DwarfStarProvider._type_id]
    for provider in ds4_providers:
        if not getattr(provider, "gguf_path", None):
            log.warning(
                f"ds4 provider #{provider._instance_id} has no gguf_path; "
                "its VRAM footprint cannot be estimated yet"
            )
            continue
        ctxs = sorted({c for c in (provider.ctx_length, default_ctx) if c})
        for ctx in ctxs:
            estimate = provider._estimate(ctx)
            log.info(
                f"ds4 VRAM estimate model={provider.gguf_path} ctx={ctx} "
                f"sessions={provider._sessions()} "
                f"projected={ds4_estimate_mib(estimate, provider._sessions()):.0f} MiB "
                f"source={estimate['source']}"
            )
