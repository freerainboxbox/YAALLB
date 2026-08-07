from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider

# ds4 cannot answer a native /v1/models while it is spawned/terminated by
# Python, so its model list is built here. Both model IDs point to the same
# underlying model; the presented context_length is 1000000 (DeepSeek v4's
# maximum) unless a model is resident with a different ctx_length.
DS4_CONTEXT_LENGTH = 1000000


class DwarfStarProvider(Provider):
    _type_id = "ds4"

    class Model(BaseModel):
        def memory(self) -> float:
            ctx = self.loadOptions.ctx_length
            if ctx >= 4224:
                return 83065.32 + 16416 * ctx / (2**20)
            return 83065.32 + 0.015655 * ctx

    def __init__(self, _instance_id: int = 0, config: dict | None = None) -> None:
        self.host = "127.0.0.1"
        self.port = 8000
        self.resident_model: BaseModel | None = None
        super().__init__(_instance_id, config)

    @property
    def endpoint_uri(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor("deepseek-v4-flash", self),
            ModelDescriptor("deepseek-v4-pro", self),
        ]

    def getOAIModels(self) -> list[dict]:
        ctx_length = DS4_CONTEXT_LENGTH
        if self.resident_model is not None:
            ctx_length = self.resident_model.loadOptions.ctx_length

        def model_entry(model_id: str) -> dict:
            return {
                "id": model_id,
                "object": "model",
                "created": 1767225600,
                "owned_by": "ds4.c",
                "name": "DeepSeek V4 Flash",
                "context_length": ctx_length,
                "top_provider": {
                    "context_length": DS4_CONTEXT_LENGTH,
                    "max_completion_tokens": 393216,
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
                    "reasoning_effort",
                ],
            }

        return [model_entry("deepseek-v4-flash"), model_entry("deepseek-v4-pro")]

    def createModel(
        self, descriptor: ModelDescriptor, loadOptions: LoadOptions
    ) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model: BaseModel) -> None:
        self.resident_model = model
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        self.resident_model = None
        model._loaded = False
