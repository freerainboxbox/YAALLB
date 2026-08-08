from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .descriptor import ModelDescriptor
    from .load_options import LoadOptions
    from .model import Model


class Provider(ABC):
    # Whether this provider can hold only a single loaded model at a time
    # (e.g. ds4). When true, any resident model from the provider serves all
    # of that provider's model IDs.
    single_resident = False

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self._instance_id = _instance_id
        self.api_key: str | None = None
        if config:
            for key, value in config.items():
                setattr(self, key, value)

    @property
    @abstractmethod
    def _type_id(self) -> str:
        """Provider type identifier, e.g. "lms" or "ds4"."""

    @property
    @abstractmethod
    def endpoint_uri(self) -> str:
        """Base URI where all reverse proxy requests are forwarded."""

    @property
    @abstractmethod
    def Model(self) -> type["Model"]:
        """The concrete Model subclass this provider creates."""

    @abstractmethod
    def getModelsDescriptors(self) -> list["ModelDescriptor"]:
        """List the descriptors of models this provider can serve."""

    def getOAIModels(self) -> list[dict]:
        """List the models this provider presents over its /v1/models.

        Default implementation queries the downstream endpoint. Override for
        providers that cannot answer that endpoint natively (e.g. ds4, which
        is spawned/terminated by Python rather than serving its own API).
        """
        resp = httpx.get(self.endpoint_uri + "/models", headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json().get("data", [])

    def _auth_headers(self) -> dict:
        if getattr(self, "api_key", None):
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    @abstractmethod
    def createModel(
        self, descriptor: "ModelDescriptor", loadOptions: "LoadOptions"
    ) -> "Model":
        """Create a Model handle for a descriptor with the given load options."""

    @abstractmethod
    def loadModel(self, model: "Model") -> None:
        """Load the given model into memory."""

    @abstractmethod
    def unloadModel(self, model: "Model") -> None:
        """Unload the given model from memory."""
