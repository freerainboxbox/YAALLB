import re
import subprocess

import log
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


def _estimate_gpu_memory(model_id: str, ctx_length: int) -> float:
    proc = subprocess.run(
        ["lms", "load", "-y", "--estimate-only", "-c", str(ctx_length), model_id],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error(
            f"lms estimate failed for {model_id} "
            f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
        raise RuntimeError(
            f"lms estimate failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    match = re.search(r"Estimated GPU Memory:\s*([\d.]+)\s*(MiB|GiB)", proc.stdout)
    if match is None:
        log.error(f"unexpected lms output for {model_id}: {proc.stdout!r}")
        raise RuntimeError(
            f"unexpected lms output, no GPU memory estimate: {proc.stdout!r}"
        )
    value = float(match.group(1))
    if match.group(2) == "GiB":
        return value * 1024
    return value


class LMStudioProvider(Provider):
    _type_id = "lms"

    class Model(BaseModel):
        def memory(self) -> float:
            return _estimate_gpu_memory(
                self.descriptor.modelId, self.loadOptions.ctx_length
            )

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = "127.0.0.1"
        self.port = 1234
        super().__init__(_instance_id, config)

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        # TODO: query LM Studio's model list endpoint.
        return []

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model: BaseModel) -> None:
        # TODO: call LM Studio's load endpoint.
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        # TODO: call LM Studio's unload endpoint.
        model._loaded = False
