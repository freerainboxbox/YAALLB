from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import Provider


def lookup_model(providers: list["Provider"], model_id: str):
    """Find the provider that serves the given model ID.

    Ties are broken in iterator order: the first provider whose descriptor
    matches is returned. Returns None when no provider serves the model.
    """
    for provider in providers:
        for descriptor in provider.getModelsDescriptors():
            if descriptor.modelId == model_id:
                return provider
    return None
