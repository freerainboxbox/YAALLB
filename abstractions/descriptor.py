from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import Provider


class ModelDescriptor:
    def __init__(self, modelId: str, provider: "Provider") -> None:
        self.modelId = modelId
        self.provider = provider
