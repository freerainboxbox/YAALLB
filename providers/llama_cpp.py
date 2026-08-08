import os
import shlex
import subprocess

import log
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider

# llama_cpp is a single-model provider: it serves one gguf_path, which maps to
# a single OAI model ID given by the mandatory `alias` config key (unlike lms,
# which can hold many models). A single resident model serves all requests.
LLAMA_CPP_DEFAULT_HOST = "127.0.0.1"
LLAMA_CPP_DEFAULT_PORT = 8080

# llama-fit-params prints one line per memory buffer to stdout, each of the
# form `<device> <model> <context> <compute>` (all MiB). The VRAM footprint is
# the sum of the three numbers across every non-Host device; Host lines are CPU
# memory and must not count toward the VRAM budget.
FIT_PARAMS_BINARY = "llama-fit-params"
SERVER_BINARY = "llama-server"

# llama-server terminate grace period before escalating to SIGKILL on unload.
UNLOAD_TERMINATE_TIMEOUT = 10.0

# Flag registry: config key -> (flag, kind, default). These are the "core
# options" — the memory-affecting flags that BOTH llama-server and
# llama-fit-params accept (the llama.cpp common params), so YAALLB passes them
# to the server on launch and to llama-fit-params for the VRAM estimate.
# `-c/--ctx-size` is deliberately absent: it comes from _effective_ctx at load
# time. On-by-default negatable flags (--no-kv-offload, --no-repack, --no-mmap)
# and server-only flags (--host, --port, --alias, --api-key, speculative/MTP
# heads, sampling) do not fit this simple pattern and live in extra_options.
LLAMA_CPP_OPTIONS = {
    "ngl": ("--gpu-layers", "value", None),
    "cache_type_k": ("--cache-type-k", "value", None),
    "cache_type_v": ("--cache-type-v", "value", None),
    "flash_attn": ("--flash-attn", "value", None),
    "swa_full": ("--swa-full", "flag", False),
    "parallel": ("--parallel", "value", None),
    "batch_size": ("-b", "value", None),
    "ubatch_size": ("-ub", "value", None),
    "split_mode": ("--split-mode", "value", None),
    "tensor_split": ("--tensor-split", "value", None),
    "main_gpu": ("--main-gpu", "value", None),
    "device": ("--device", "value", None),
    "override_tensor": ("--override-tensor", "value", None),
    "cpu_moe": ("--cpu-moe", "flag", False),
    "n_cpu_moe": ("--n-cpu-moe", "value", None),
    "load_mode": ("--load-mode", "value", None),
    "no_host": ("--no-host", "flag", False),
    "lora": ("--lora", "value", None),
    "lora_scaled": ("--lora-scaled", "value", None),
    "control_vector": ("--control-vector", "value", None),
    "control_vector_scaled": ("--control-vector-scaled", "value", None),
    "threads": ("-t", "value", None),
    "threads_batch": ("-tb", "value", None),
    "mlock": ("--mlock", "flag", False),
    "rope_scaling": ("--rope-scaling", "value", None),
    "rope_scale": ("--rope-scale", "value", None),
    "rope_freq_base": ("--rope-freq-base", "value", None),
    "rope_freq_scale": ("--rope-freq-scale", "value", None),
    "yarn_orig_ctx": ("--yarn-orig-ctx", "value", None),
    "yarn_ext_factor": ("--yarn-ext-factor", "value", None),
    "yarn_attn_factor": ("--yarn-attn-factor", "value", None),
    "yarn_beta_slow": ("--yarn-beta-slow", "value", None),
    "yarn_beta_fast": ("--yarn-beta-fast", "value", None),
    "cache_type_k_draft": ("--cache-type-k-draft", "value", None),
    "cache_type_v_draft": ("--cache-type-v-draft", "value", None),
    "rpc": ("--rpc", "value", None),
}


def _options_flags(options: dict) -> list[str]:
    """Emit argv tokens for the configured registry options (ds4-style)."""
    flags = []
    for key, (flag, kind, default) in LLAMA_CPP_OPTIONS.items():
        if key not in options:
            continue
        value = options[key]
        if kind == "flag":
            if value is True:
                flags.append(flag)
        elif value != default:
            flags += [flag, str(value)]
    return flags


def _fit_memory_mib(
    llama_cpp_dir: str, gguf_path: str, ctx_length: int, options: dict
) -> float:
    if llama_cpp_dir is None:
        raise ValueError("llama_cpp_dir must be set in config before estimating memory")
    if gguf_path is None:
        raise ValueError("gguf_path must be set in config before estimating memory")
    binary = os.path.join(llama_cpp_dir, FIT_PARAMS_BINARY)
    command = [
        binary, "-m", gguf_path, "--ctx-size", str(ctx_length), "--fit-print", "on",
    ]
    command += _options_flags(options)
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=llama_cpp_dir,
    )
    if proc.returncode != 0:
        log.error(
            f"llama-fit-params failed for {gguf_path} "
            f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
        raise RuntimeError(
            f"llama-fit-params failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    total = 0.0
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "Host":
            continue
        try:
            total += sum(float(x) for x in parts[1:4])
        except ValueError:
            # A header/annotation line whose <model>/<context>/<compute>
            # columns are non-numeric must fail loudly as a RuntimeError, not
            # escape as a raw ValueError that bypasses the fail-fast contract.
            raise RuntimeError(
                f"unexpected non-numeric llama-fit-params column for {gguf_path}: "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            ) from None
    if total <= 0:
        raise RuntimeError(
            f"unexpected llama-fit-params output for {gguf_path}: "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return total


class LlamaCppProvider(Provider):
    _type_id = "llama_cpp"
    single_resident = True

    class Model(BaseModel):
        def memory(self) -> float:
            ctx = self.descriptor.provider._effective_ctx(self)
            return _fit_memory_mib(
                self.descriptor.provider.llama_cpp_dir,
                self.descriptor.provider.gguf_path,
                ctx,
                self.descriptor.provider.options,
            )

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = LLAMA_CPP_DEFAULT_HOST
        self.port = LLAMA_CPP_DEFAULT_PORT
        self.llama_cpp_dir: str | None = None
        self.gguf_path: str | None = None
        self.ctx_length: int | None = None
        self.alias: str | None = None
        self.extra_options: str | None = None
        self.options: dict = {}
        self.resident_model: BaseModel | None = None
        self._process = None
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
        # llama_cpp serves exactly one model (the mandatory `alias`), so its
        # descriptor list is a single entry keyed on the alias.
        return [ModelDescriptor(self.alias, self)]

    def getOAIModels(self) -> list[dict]:
        ctx_length = self._effective_ctx(self.resident_model)
        return [
            {
                "id": self.alias,
                "object": "model",
                "created": 1767225600,
                "owned_by": "llama.cpp",
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

    def _build_command(self, model: BaseModel) -> list[str]:
        ctx_length = self._effective_ctx(model)

        command = [os.path.join(self.llama_cpp_dir, SERVER_BINARY), "-m", self.gguf_path]

        if self.host != LLAMA_CPP_DEFAULT_HOST:
            command += ["--host", self.host]
        if self.port != LLAMA_CPP_DEFAULT_PORT:
            command += ["--port", str(self.port)]
        if self.alias is not None:
            command += ["-a", self.alias]

        command += _options_flags(self.options)

        command += ["-c", str(ctx_length)]

        if self.extra_options:
            # Inserted verbatim last, after YAALLB's own flags AND its -c, so
            # extra_options can override every YAALLB default (including a
            # different -c). Tokenized with shlex (no shell), so quoted values
            # and paths with spaces survive; paths are resolved by llama-server
            # against cwd=llama_cpp_dir.
            command += shlex.split(self.extra_options)

        return command

    def loadModel(self, model: BaseModel) -> None:
        if self.llama_cpp_dir is None:
            raise ValueError("llama_cpp_dir must be set in config before loading")
        if self.gguf_path is None:
            raise ValueError("gguf_path must be set in config before loading")

        command = self._build_command(model)
        self._process = subprocess.Popen(command, cwd=self.llama_cpp_dir)
        self.resident_model = model
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=UNLOAD_TERMINATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                # wait(timeout) raises instead of returning, so a llama-server
                # that ignores SIGTERM (stuck in a Metal kernel or KV disk I/O)
                # must be escalated to SIGKILL here; letting the exception
                # escape would wedge the scheduler's eviction path.
                process.kill()
                process.wait()
            self._process = None
        self.resident_model = None
        model._loaded = False
