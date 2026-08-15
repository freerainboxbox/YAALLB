import subprocess
import time

import httpx
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider

# ds4 cannot answer a native /v1/models while it is spawned/terminated by
# Python, so its model list is built here. Both model IDs point to the same
# underlying model; the presented context_length is 1000000 (DeepSeek v4's
# maximum) unless a model is resident with a different ctx_length.
DS4_CONTEXT_LENGTH = 1000000

DS4_DEFAULT_HOST = "127.0.0.1"
DS4_DEFAULT_PORT = 8000
DS4_DEFAULT_BINARY = "./ds4-server"

# How long loadModel waits for the spawned ds4-server to start accepting
# requests before failing the load. ds4 loads the model at launch, so a 200
# from /v1/models means the model is resident and ready.
DS4_READY_TIMEOUT = 120
READY_POLL_INTERVAL = 0.5


# Wait for a spawned server to start accepting requests on its endpoint.
# Blocks (off the event loop; loadModel runs in a worker thread) until GET
# {endpoint}/models returns 200, the process exits, or the timeout elapses.
# Used so `model.loaded` reflects the downstream being actually ready, not
# just the process having been spawned.
def _wait_server_ready(endpoint_uri: str, process, label: str) -> None:
    deadline = time.monotonic() + DS4_READY_TIMEOUT
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} exited before becoming ready")
        try:
            resp = httpx.get(endpoint_uri + "/models")
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{label} not ready within {DS4_READY_TIMEOUT}s")
        time.sleep(READY_POLL_INTERVAL)

# Flag registry: config key -> (flag, kind, default). Defaults grabbed from
# `./ds4-server --help`. `--ctx` is deliberately absent: it comes from
# LoadOptions.ctx_length at load time.
DS4_OPTIONS = {
    "backend": ("--backend", "value", None),
    "metal": ("--metal", "flag", False),
    "cuda": ("--cuda", "flag", False),
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
    "cors": ("--cors", "flag", False),
    "trace": ("--trace", "value", None),
    "batched_session": ("--batched-session", "value", None),
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
            ctx = self.descriptor.provider._effective_ctx()
            if ctx >= 4224:
                return 83065.32 + 16416 * ctx / (2**20)
            return 83065.32 + 0.015655 * ctx

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = DS4_DEFAULT_HOST
        self.port = DS4_DEFAULT_PORT
        self.ds4_dir: str | None = None
        self.gguf_path: str | None = None
        self.binary: str = DS4_DEFAULT_BINARY
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
            _wait_server_ready(self.endpoint_uri, self._process, "ds4-server")
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
