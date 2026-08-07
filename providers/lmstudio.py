from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


class LMStudioProvider(Provider):
    _type_id = "lms"

    class Model(BaseModel):
        def memory(self) -> float:
            # TODO: project footprint from LM Studio model metadata.
            return 0.0

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
