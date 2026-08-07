from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .descriptor import ModelDescriptor
    from .load_options import LoadOptions
    from .model import Model


class Provider(ABC):
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
        resp = httpx.get(self.endpoint_uri + "/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

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
