import json
import os
import signal
import subprocess

import log
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from abstractions.ready import wait_server_ready

# dflash-mlx is a CLI/HTTP server (not an embeddable API), so — consistent with
# llama_cpp and dwarfstar — YAALLB spawns it as a subprocess and terminates it
# on unload. It serves one target (+optional draft) per process.
DFLASH_BINARY = "dflash"
DFLASH_DEFAULT_HOST = "127.0.0.1"
DFLASH_DEFAULT_PORT = 8000

# dflash waits for the model to load BEFORE it starts listening
# (serve_forever calls wait_until_ready() then _run_http_server), so a 200
# from GET {endpoint}/models means the model is resident AND ready.
DFLASH_READY_TIMEOUT = 120

# Context length presented by /v1/models before any model is resident (dflash
# is spawned lazily on load, so it cannot answer /v1/models at startup).
DFLASH_DEFAULT_CTX = 4096

# dflash's own teardown (L2 cache flush via shutdown_runtime_cache_manager)
# only runs on SIGINT — mlx_lm._run_http_server catches KeyboardInterrupt.
# So unloadModel prefers SIGINT first (clean flush), escalating to
# SIGTERM/SIGKILL on timeout.
UNLOAD_TERMINATE_TIMEOUT = 10.0
UNLOAD_SIGTERM_TIMEOUT = 5.0

# Flag registry for `dflash serve` (dflash_mlx/server/config.py build_parser):
# config key -> (flag, kind, default). Memory-affecting flags only (the
# estimate stays in sync with the actual configuration): metal limits, draft
# quant, KV-cache quant, and the draft cache structure sizes. --wired-limit
# default is the string "auto" (metavar auto|none|BYTES).
DFLASH_OPTIONS = {
    "wired_limit": ("--wired-limit", "value", "auto"),
    "cache_limit": ("--cache-limit", "value", None),
    "draft_quant": ("--draft-quant", "value", None),
    "quantize_kv_cache": ("--quantize-kv-cache", "flag", False),
    "draft_sink_size": ("--draft-sink-size", "value", 64),
    "draft_window_size": ("--draft-window-size", "value", 1024),
    "draft_full_context_min_ctx": ("--draft-full-context-min-ctx", "value", 16384),
}


def _options_flags(options: dict) -> list[str]:
    flags = []
    for key, (flag, kind, default) in DFLASH_OPTIONS.items():
        if key not in options:
            continue
        value = options[key]
        if kind == "flag":
            if value is True:
                flags.append(flag)
        elif value != default:
            flags += [flag, str(value)]
    return flags


def _read_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def _model_path(provider) -> str:
    """Resolved target dir: the startup-resolved path, else resolve on the fly."""
    path = getattr(provider, "model_path", None)
    if path:
        return path
    from providers.dflash_shortcuts import resolve_model_path

    return resolve_model_path(provider.model_ref)


def _projected_from_cache(
    entry: dict, provider, model: BaseModel
) -> float:
    """projected_mib from cached ctx-independent components + ctx-scaled KV."""
    from providers.dflash_vram import WORKING_SET_OVERHEAD, target_kv_bytes

    ctx = provider._effective_ctx(model)
    target_config = _read_config(_model_path(provider))
    target_kv = target_kv_bytes(target_config, ctx)
    base = (
        entry["target_weight_bytes"]
        + entry["draft_weight_bytes"]
        + target_kv
        + entry["draft_kv_bytes"]
        + entry["draft_context_bytes"]
    )
    return base * WORKING_SET_OVERHEAD / (2**20)


def _projected_from_engine(provider, model: BaseModel) -> float:
    """Defensive recompute when the model is missing from the VRAM cache."""
    from providers.dflash_cache import compute_impact
    from providers.dflash_vram import compute_weights as engine

    model_path = _model_path(provider)
    draft_path = getattr(provider, "draft_path", None)
    if draft_path is None:
        eff = getattr(provider, "draft_ref_eff", None) or getattr(
            provider, "draft_ref", None
        )
        if eff:
            from providers.dflash_shortcuts import resolve_model_path

            draft_path = resolve_model_path(eff)
    impact = compute_impact(model_path, draft_path, engine)
    return _projected_from_cache(impact, provider, model)


class DflashProvider(Provider):
    _type_id = "dflash-mlx"
    single_resident = True

    class Model(BaseModel):
        def memory(self) -> float:
            provider = self.descriptor.provider
            cache = getattr(provider, "_vram_cache", None)
            model_id = self.descriptor.modelId
            if cache is not None:
                entry = cache.get("dflash-mlx", {}).get(model_id)
                if entry is not None:
                    return _projected_from_cache(entry, provider, self)
                log.warning(
                    f"model {model_id} missing from dflash VRAM cache; recomputing"
                )
            # Defensive: cache miss / not populated -> compute on the fly.
            return _projected_from_engine(provider, self)

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = DFLASH_DEFAULT_HOST
        self.port = DFLASH_DEFAULT_PORT
        self.dflash_dir: str | None = None
        self.model_ref: str | None = None
        self.draft_ref: str | None = None
        self.binary: str = DFLASH_BINARY
        self.alias: str | None = None
        self.ctx_length: int | None = None
        self.options: dict = {}
        self.resident_model: BaseModel | None = None
        self._process: subprocess.Popen | None = None
        # Resolved local paths (set by ensure_dflash_cache at startup):
        # model_path/draft_path are local dirs (local or HF-cache snapshot);
        # draft_ref_eff is the effective draft ref (shortcut default if none).
        self.model_path: str | None = None
        self.draft_path: str | None = None
        self.draft_ref_eff: str | None = None
        super().__init__(_instance_id, config)

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _effective_ctx(self, model: BaseModel | None = None) -> int:
        # A provider-level ctx_length (when set) overrides any per-model one.
        # A loading model is passed explicitly: at load time resident_model is
        # still None, so without it the fallback would miss the request's ctx.
        if self.ctx_length is not None:
            return self.ctx_length
        if model is not None:
            return model.loadOptions.ctx_length
        if self.resident_model is not None:
            return self.resident_model.loadOptions.ctx_length
        raise ValueError(
            "ctx_length unknown: set provider-level ctx_length or pass a model"
        )

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        # dflash serves exactly one target+draft per process, presented under
        # the mandatory `alias` config key (like llama_cpp).
        return [ModelDescriptor(self.alias, self)]

    def _presented_ctx(self) -> int:
        # Context length shown by /v1/models: provider ctx_length, else the
        # resident model's, else a default (dflash is lazily spawned, so it
        # cannot answer /v1/models itself at startup).
        if self.ctx_length is not None:
            return self.ctx_length
        if self.resident_model is not None:
            return self.resident_model.loadOptions.ctx_length
        return DFLASH_DEFAULT_CTX

    def getOAIModels(self) -> list[dict]:
        # dflash is spawned lazily on load (not at startup), so the default
        # Provider.getOAIModels() (an HTTP query) would hit a non-running
        # server and 500 /v1/models. Present a static single-model list keyed
        # on the configured `alias`, like llama_cpp.
        ctx_length = self._presented_ctx()
        return [
            {
                "id": self.alias,
                "object": "model",
                "created": 1767225600,
                "owned_by": "dflash-mlx",
                "name": self.alias,
                "context_length": ctx_length,
                "top_provider": {
                    "context_length": ctx_length,
                    "max_completion_tokens": ctx_length,
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
                ],
            }
        ]

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def _binary_path(self) -> str:
        """Resolve the ``dflash`` binary to a runnable path.

        The bare default ``dflash`` is resolved via PATH; dflash-mlx run from
        its repo via ``uv`` keeps the console script at ``<dflash_dir>/.venv/
        bin/dflash`` (not on PATH), so also probe that layout before falling
        back to the bare name. Absolute ``binary`` is used verbatim.
        """
        binary = self.binary
        if os.path.isabs(binary):
            return binary
        if self.dflash_dir:
            for candidate in (
                os.path.join(self.dflash_dir, binary),
                os.path.join(self.dflash_dir, ".venv", "bin", binary),
            ):
                if os.path.exists(candidate):
                    return candidate
        return binary

    def _build_command(self, model: BaseModel) -> list[str]:
        # Spawn with the resolved local paths (HF-cache snapshot or local dir)
        # so dflash does not re-resolve/download; fall back to the configured
        # refs if resolution did not run (e.g. tests without startup).
        command = [self._binary_path(), "serve", "--model", self.model_path or self.model_ref]

        if self.host != DFLASH_DEFAULT_HOST:
            command += ["--host", self.host]
        if self.port != DFLASH_DEFAULT_PORT:
            command += ["--port", str(self.port)]
        draft = self.draft_path or getattr(self, "draft_ref_eff", None) or self.draft_ref
        if draft:
            command += ["--draft-model", draft]

        command += _options_flags(self.options)
        return command

    def loadModel(self, model: BaseModel) -> None:
        if self.dflash_dir is None:
            raise ValueError("dflash_dir must be set in config before loading")
        if self.model_ref is None:
            raise ValueError("model_ref must be set in config before loading")

        command = self._build_command(model)
        self._process = subprocess.Popen(command, cwd=self.dflash_dir)
        self.resident_model = model
        # The model is loading until the spawned server actually accepts
        # requests; only then is it marked loaded (ready).
        model._loaded = False
        model._load_state = "loading"
        try:
            wait_server_ready(
                self.endpoint_uri, self._process, "dflash", DFLASH_READY_TIMEOUT
            )
        except Exception:
            # A readiness timeout (or server exit) must not orphan the spawned
            # dflash: terminate it and clear provider state before the load
            # fails, so a VRAM-holding child isn't leaked.
            self.unloadModel(model)
            raise
        model._loaded = True
        model._load_state = "ready"

    def unloadModel(self, model: BaseModel) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                # SIGINT first: KeyboardInterrupt -> httpd.shutdown() +
                # L2 cache flush (dflash's only clean teardown path).
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=UNLOAD_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.terminate()  # SIGTERM escalation
                    try:
                        process.wait(timeout=UNLOAD_SIGTERM_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        # SIGKILL escalation (stuck in Metal kernel / disk I/O)
                        process.kill()
                        process.wait()
            self._process = None
        self.resident_model = None
        model._loaded = False
