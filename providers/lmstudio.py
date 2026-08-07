import re
import subprocess
import time

import httpx

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


# How long model descriptors stay cached before re-querying the server.
_DESCRIPTORS_TTL = 30.0


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
        self._descriptors: list[ModelDescriptor] | None = None
        self._descriptors_at: float = 0.0
        super().__init__(_instance_id, config)

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _rest_uri(self) -> str:
        # LM Studio's management API (load/unload/list) lives under /api/v1,
        # separate from the OpenAI-compatible /v1 surface.
        return f"http://{self.host}:{self.port}/api/v1"

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        now = time.monotonic()
        if self._descriptors is None or now - self._descriptors_at > _DESCRIPTORS_TTL:
            resp = httpx.get(
                self._rest_uri() + "/models", headers=self._auth_headers()
            )
            resp.raise_for_status()
            self._descriptors = [
                ModelDescriptor(m["key"], self)
                for m in resp.json().get("models", [])
                if m.get("type") == "llm"
            ]
            self._descriptors_at = now
        return list(self._descriptors)

    def getOAIModels(self) -> list[dict]:
        return super().getOAIModels()

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model: BaseModel) -> None:
        resp = httpx.post(
            self._rest_uri() + "/models/load",
            json={
                "model": model.descriptor.modelId,
                "context_length": model.loadOptions.ctx_length,
            },
            headers=self._auth_headers(),
        )
        if resp.status_code != 200:
            log.error(
                f"lms load failed for {model.descriptor.modelId} "
                f"(status {resp.status_code}): {resp.text}"
            )
            raise RuntimeError(
                f"lms load failed (status {resp.status_code}): {resp.text}"
            )
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        resp = httpx.post(
            self._rest_uri() + "/models/unload",
            json={"instance_id": model.descriptor.modelId},
            headers=self._auth_headers(),
        )
        if resp.status_code != 200:
            log.error(
                f"lms unload failed for {model.descriptor.modelId} "
                f"(status {resp.status_code}): {resp.text}"
            )
            raise RuntimeError(
                f"lms unload failed (status {resp.status_code}): {resp.text}"
            )
        model._loaded = False
