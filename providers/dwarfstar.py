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
    merged_env,
)
from providers.ds4_models import (
    DS4_DEFAULT_MAX_COMPLETION_TOKENS,
    DS4_DEFAULT_PROFILE,
    DS4_MODEL_PROFILES,
    DS4_SUPPORTED_PARAMETERS,
    Ds4ModelProfile,
    profile_for,
    profile_named,
)

# ds4 cannot answer a native /v1/models while it is spawned/terminated by
# Python, so the model list is built here from the profile registry (see
# providers/ds4_models.py). Every alias of the served family names the same
# resident model, and the presented context_length is the ctx that will actually
# be spawned. This is the context used for a family ds4 gives no documented
# ceiling of its own (and is what DeepSeek V4 has always been given here).
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
    # Directional steering: loaded per layer as f32 vectors, small enough that it
    # does not need its own footprint term (use safety_buffer_mib if your vectors
    # are unusually large). ds4 defaults --dir-steering-ffn to 1 when a file is
    # given without a scale, so the scales are passed whenever configured.
    "dir_steering_file": ("--dir-steering-file", "value", None),
    "dir_steering_ffn": ("--dir-steering-ffn", "value", None),
    "dir_steering_attn": ("--dir-steering-attn", "value", None),
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
        # ds4 reads a handful of knobs from the environment only, and several of
        # them change the footprint (static YaRN resizes what a context costs), so
        # they go to the server and to the estimator alike.
        self.env: dict | None = None
        # How long to wait for a spawned ds4-server to answer. Loading a 165 GiB
        # GGUF takes far longer than a small model, and this provider has no
        # native load/unload to report progress through.
        self.ready_timeout: int | None = None
        # Which ds4 family this instance serves (see providers/ds4_models.py).
        # None means "whatever the GGUF is", which is confirmed against the ds4
        # build itself when its footprint is estimated.
        self.model_profile: str | None = None
        self.resident_model: BaseModel | None = None
        self._process: subprocess.Popen | None = None
        # What the opened GGUF turned out to be, learned from ds4 itself the
        # first time its estimator runs (see _adopt_identity).
        self._detected_profile: Ds4ModelProfile | None = None
        self._model_name: str | None = None
        self._identity_adopted = False
        super().__init__(_instance_id, config)
        self._explicit_profile = self._resolve_profile()

    @property
    def served_profile(self) -> Ds4ModelProfile:
        """Configured profile, else the one ds4 confirmed, else the old default."""
        return self._explicit_profile or self._detected_profile or DS4_DEFAULT_PROFILE

    def _resolve_profile(self) -> Ds4ModelProfile:
        """The configured model profile, or the family YAALLB always assumed.

        A wrong `model_profile` is a config error and says so at startup, with
        the names it accepts. ds4's own flag combinations are not checked
        anywhere: what a model rejects, it logs itself.
        """
        if self.model_profile is None:
            return None
        profile = profile_named(self.model_profile)
        if profile is None:
            accepted = ", ".join(
                f"{p.family} ({p.primary_alias})" for p in DS4_MODEL_PROFILES
            )
            raise ValueError(
                f"model_profile {self.model_profile!r} is not a ds4 model "
                f"profile; use one of: {accepted} (or any of their aliases)"
            )
        return profile

    def _adopt_identity(self, estimate: dict) -> None:
        """Record which model ds4 says the configured GGUF actually is.

        The GGUF - not config.json - decides what a ds4 tree serves, and only ds4
        knows which shape it read out of the file. So the first estimator result
        that carries a shape teaches this provider its model IDs, its display
        name and its context ceiling, which is what stops a Qwen3.8 tree from
        being presented as DeepSeek V4 Flash with a million tokens of context.

        An estimator that fell back to GGUF file sizes carries no shape, so it
        does not consume this one chance to learn it.
        """
        if self._identity_adopted:
            return
        if estimate.get("model_name") is None and estimate.get("model_family") is None:
            return
        self._identity_adopted = True

        self._model_name = estimate.get("model_name") or self._model_name
        family = estimate.get("model_family")

        if self.model_profile is not None:
            # An explicit profile is an override and stays one either way; when
            # config and GGUF disagree, that is worth reading in the log.
            if self.served_profile.family == family:
                log.info(
                    f"ds4 provider #{self._instance_id} serves "
                    f"model_profile={self.model_profile}, confirmed by ds4 "
                    f"(name={estimate.get('model_name')!r})"
                )
            else:
                log.info(
                    f"ds4 provider #{self._instance_id} serves model_profile="
                    f"{self.model_profile} although ds4 reports family={family!r} "
                    f"name={estimate.get('model_name')!r}"
                )
            return

        profile = profile_for(family)
        if profile is None:
            log.warning(
                f"ds4 provider #{self._instance_id} opened a GGUF ds4 reports as "
                f"family={family!r} ({estimate.get('model_name')!r}), which "
                "YAALLB has no profile for; presenting "
                f"{DS4_DEFAULT_PROFILE.primary_alias} and its aliases instead. "
                "Set model_profile explicitly, or add the family to "
                "providers/ds4_models.py."
            )
            return

        self._detected_profile = profile
        log.info(
            f"ds4 provider #{self._instance_id} serves {profile.family} "
            f"({estimate.get('model_name')}) as {len(profile.aliases)} model ids"
        )

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _effective_ctx(self, model: BaseModel | None = None) -> int:
        # ds4 sets --ctx once at startup and every served alias inherits it, so
        # the provider-level ctx_length (when set) overrides any per-model one.
        # A loading model is passed explicitly: at spawn time resident_model is
        # still None, so without it the fallback context would be spawned while
        # memory()//v1/models account the request's ctx.
        if self.ctx_length is not None:
            return self.ctx_length
        if model is not None:
            return model.loadOptions.ctx_length
        if self.resident_model is not None:
            return self.resident_model.loadOptions.ctx_length
        return self._default_ctx()

    def _default_ctx(self) -> int:
        """The context used when nothing asked for one: the family's own.

        A GGUF decides which model a ds4 tree serves, but --ctx still comes from
        here, so a shape built for one context must not be spawned at another
        family's number: Qwen3.8 Flash Next is sized at 262144, while DeepSeek
        V4 keeps the million this provider hardcoded before any registry
        existed. Families ds4 gives no documented ceiling keep that same
        long-standing value rather than gaining an invented one.
        """
        return self.served_profile.native_ctx or DS4_CONTEXT_LENGTH

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
        estimate = self._estimate_uncached(ctx)
        # The run that prices the model also says what it is. The cheapest
        # context is estimated first, so the shape is known before the big ones.
        self._adopt_identity(estimate)
        return estimate

    def _estimate_uncached(self, ctx: int) -> dict:
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
            # Qwen3.8 Flash Next and GLM 5.3 carry their MTP block in the main
            # GGUF, so --mtp (not a support-file path) is what says a drafter is
            # configured, and ds4's mtp_draft_tokens only answer with it.
            mtp=bool(self.options.get("mtp", False)),
            # Only a family named in config is available to guide a fallback
            # estimate: the estimator is the other source, and it may be the
            # thing that just failed.
            model_family=self._configured_family(),
            # The same environment the server will run under: ds4's own knobs
            # change what a context costs, so estimating under a different one
            # would price a configuration it is not asked to run.
            extra_env=self.env,
        )

    def _configured_family(self) -> str | None:
        return self._explicit_profile.family if self._explicit_profile else None

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        # Every alias of the served family is registered, because each is a
        # request YAALLB has to route (and budget) - and the thinking aliases
        # only work if clients can name them.
        return [ModelDescriptor(alias, self) for alias in self.served_profile.aliases]

    def _display_name(self) -> str:
        # ds4's own shape name: it tells Flash from PRO (one profile, one family)
        # and reports renamed shapes like Vision Experimental.
        return self._model_name or self.served_profile.display_name

    def getOAIModels(self) -> list[dict]:
        # Mirrors ds4-server's own /v1/models (ds4_server.c append_model_json):
        # the server ctx in both context fields, its -n as the completion
        # ceiling, one supported_parameters list for every model it serves.
        ctx_length = self._effective_ctx()
        max_completion = int(
            self.options.get("tokens") or DS4_DEFAULT_MAX_COMPLETION_TOKENS
        )

        def model_entry(model_id: str) -> dict:
            return {
                "id": model_id,
                "object": "model",
                "created": 1767225600,
                "owned_by": "ds4.c",
                "name": self._display_name(),
                "context_length": ctx_length,
                "top_provider": {
                    "context_length": ctx_length,
                    "max_completion_tokens": max_completion,
                    "is_moderated": False,
                },
                "supported_parameters": DS4_SUPPORTED_PARAMETERS,
            }

        return [model_entry(alias) for alias in self.served_profile.aliases]

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
        self._process = subprocess.Popen(
            command,
            cwd=self.ds4_dir,
            # None when nothing is configured, so the child simply inherits.
            env=merged_env(self.env),
        )
        self.resident_model = model
        # The model is loading until the spawned server actually accepts
        # requests; only then is it marked loaded (ready).
        model._loaded = False
        model._load_state = "loading"
        try:
            wait_server_ready(
                self.endpoint_uri,
                self._process,
                "ds4-server",
                # Configured per instance: a 165 GiB GGUF boots far outside the
                # timeout a small model needs.
                self.ready_timeout or DS4_READY_TIMEOUT,
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
                f"family={provider.served_profile.family} "
                f"sessions={provider._sessions()} "
                f"projected={ds4_estimate_mib(estimate, provider._sessions()):.0f} MiB "
                f"source={estimate['source']}"
            )
