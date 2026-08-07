import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model
from abstractions.provider import Provider
from providers.dwarfstar import DwarfStarProvider
from providers.lmstudio import LMStudioProvider


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        Provider()


def test_model_is_abstract():
    with pytest.raises(TypeError):
        Model(ModelDescriptor("m", LMStudioProvider()), LoadOptions())


def test_provider_model_is_mandatory_subclass():
    assert issubclass(LMStudioProvider.Model, Model)
    assert issubclass(DwarfStarProvider.Model, Model)


def test_descriptor_holds_model_id_and_provider():
    provider = LMStudioProvider()
    desc = ModelDescriptor("model-x", provider)
    assert desc.modelId == "model-x"
    assert desc.provider is provider


def test_load_options_default_and_extra_attrs():
    opts = LoadOptions()
    assert opts.ctx_length == 4096
    opts2 = LoadOptions(ctx_length=8192, top_p=0.9, temperature=0.2)
    assert opts2.ctx_length == 8192
    assert opts2.top_p == 0.9
    assert opts2.temperature == 0.2


def test_create_model_returns_provider_model():
    provider = LMStudioProvider()
    desc = ModelDescriptor("m", provider)
    model = provider.createModel(desc, LoadOptions())
    assert isinstance(model, provider.Model)
    assert model.descriptor is desc
    assert model.loadOptions is not None


def test_load_and_unload_alias_to_provider():
    provider = LMStudioProvider()
    desc = ModelDescriptor("m", provider)
    model = provider.createModel(desc, LoadOptions())
    assert not model.loaded
    model.load()
    assert model.loaded
    model.unloadModel()
    assert not model.loaded


def test_providers_concrete_memory():
    p1 = LMStudioProvider()
    m1 = p1.createModel(ModelDescriptor("a", p1), LoadOptions())
    assert isinstance(m1.memory(), float)
    p2 = DwarfStarProvider()
    m2 = p2.createModel(ModelDescriptor("b", p2), LoadOptions())
    assert isinstance(m2.memory(), float)


def test_providers_list_descriptors():
    assert LMStudioProvider().getModelsDescriptors() == []
    provider = DwarfStarProvider()
    descs = provider.getModelsDescriptors()
    assert [d.modelId for d in descs] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert all(d.provider is provider for d in descs)


def test_dwarfstar_memory_piecewise():
    p = DwarfStarProvider()
    small = p.createModel(ModelDescriptor("b", p), LoadOptions(ctx_length=4096))
    assert small.memory() == pytest.approx(83065.32 + 0.015655 * 4096)
    big = p.createModel(ModelDescriptor("b", p), LoadOptions(ctx_length=8192))
    assert big.memory() == pytest.approx(83065.32 + 16416 * 8192 / (2**20))
