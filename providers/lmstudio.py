from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


class LMStudioProvider(Provider):
    class Model(BaseModel):
        def memory(self) -> float:
            # TODO: project footprint from LM Studio model metadata.
            return 0.0

    def __init__(self, host: str = "127.0.0.1", port: int = 1234) -> None:
        self.host = host
        self.port = port

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
