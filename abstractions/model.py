from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .descriptor import ModelDescriptor
    from .load_options import LoadOptions


class Model(ABC):
    def __init__(
        self, descriptor: "ModelDescriptor", loadOptions: "LoadOptions"
    ) -> None:
        self.descriptor = descriptor
        self.loadOptions = loadOptions
        self._loaded = False
        # Track load progress: "loading" while a spawned/remote provider is
        # still becoming ready, "ready" once it accepts requests, "failed"
        # on a load/readiness error.
        self._load_state = "loading"

    def load(self) -> None:
        self.descriptor.provider.loadModel(self)

    def unloadModel(self) -> None:
        self.descriptor.provider.unloadModel(self)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_state(self) -> str:
        return self._load_state

    @abstractmethod
    def memory(self) -> float:
        """Projected memory footprint of the model in MiB."""
