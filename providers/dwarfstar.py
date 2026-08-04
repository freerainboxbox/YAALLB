from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


class DwarfStarProvider(Provider):
    class Model(BaseModel):
        def memory(self) -> float:
            # TODO: project footprint from ds4 metadata.
            return 0.0

    def __init__(self, host: str = "127.0.0.1", port: int = 4343) -> None:
        self.host = host
        self.port = port

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        # TODO: query ds4's model list endpoint.
        return []

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model: BaseModel) -> None:
        # TODO: call ds4's load endpoint.
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        # TODO: call ds4's unload endpoint.
        model._loaded = False
