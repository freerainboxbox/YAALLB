"""ds4's model registry (providers/ds4_models.py) and what a provider serves.

The point is that YAALLB registers the model IDs a ds4 tree really answers to -
including the aliases that turn thinking on and off - and reports them the way
ds4-server itself would, rather than assuming every ds4 tree is DeepSeek V4
Flash/PRO. The registry itself is a copy of ds4_server.c's alias tables, so the
families that matter are asserted literally: a silent rename should fail here,
not in somebody's config.
"""

import pytest

from providers.dwarfstar import DwarfStarProvider
from providers.ds4_models import (
    DS4_DEFAULT_MAX_COMPLETION_TOKENS,
    DS4_SUPPORTED_PARAMETERS,
    profile_for,
    profile_named,
)

DEEPSEEK_V4_IDS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
]


def _provider(**config):
    return DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf", **config}
    )


@pytest.fixture
def no_network(monkeypatch):
    def stop(url):
        raise AssertionError(f"should not hit the network: {url}")

    monkeypatch.setattr("httpx.get", stop)


def test_deepseek_aliases_are_all_registered(no_network):
    provider = _provider()

    # deepseek-chat / deepseek-reasoner are how a client asks ds4 for answers
    # without (or with) thinking; they used to be unroutable through YAALLB.
    assert [d.modelId for d in provider.getModelsDescriptors()] == DEEPSEEK_V4_IDS
    assert [m["id"] for m in provider.getOAIModels()] == DEEPSEEK_V4_IDS


def test_aliases_all_name_one_resident_model(no_network):
    provider = _provider()
    descriptors = provider.getModelsDescriptors()

    # ds4 serves every alias of the loaded GGUF from one process, so the
    # provider stays single-resident and the aliases share that one resident
    # model instead of competing for VRAM.
    assert DwarfStarProvider.single_resident is True
    assert all(d.provider is provider for d in descriptors)


def test_oai_listing_mirrors_ds4s_own_model_json(no_network):
    provider = _provider(options={"tokens": 4096})
    entry = provider.getOAIModels()[0]

    # ds4 prints its server ctx in both context fields and its -n default as
    # max_completion_tokens; a client comparing the two must not see YAALLB
    # inventing a bigger window than the server allocated.
    assert entry["context_length"] == 1000000
    assert entry["top_provider"]["context_length"] == 1000000
    assert entry["top_provider"]["max_completion_tokens"] == 4096
    assert entry["supported_parameters"] == DS4_SUPPORTED_PARAMETERS
    assert "ignore_eos" in entry["supported_parameters"]  # ds4 answers it
    assert entry["owned_by"] == "ds4.c"

    # Without -n configured, what ds4 defaults to is what gets reported.
    plain = _provider().getOAIModels()[0]
    assert (
        plain["top_provider"]["max_completion_tokens"]
        == DS4_DEFAULT_MAX_COMPLETION_TOKENS
    )


def test_profile_is_keyed_by_ds4s_family_name():
    assert profile_for("deepseek4").primary_alias == "deepseek-v4-flash"
    assert profile_for("no-such-family") is None
    assert profile_for(None) is None


def test_model_profile_selects_a_family_by_key_or_alias():
    assert profile_named("deepseek4") is profile_for("deepseek4")
    assert (
        profile_named("deepseek-reasoner").family
        == profile_for("deepseek4").family
    )
    assert profile_named(None) is None


def test_model_profile_in_config_selects_the_descriptors(no_network):
    provider = _provider(model_profile="deepseek4")
    assert [d.modelId for d in provider.getModelsDescriptors()] == DEEPSEEK_V4_IDS


def test_unknown_model_profile_is_rejected(no_network):
    # This is YAALLB's own config being wrong, so it says so; ds4's own flag
    # combinations are deliberately not second-guessed.
    # Raised while the provider is built, i.e. at startup, not on first request.
    with pytest.raises(ValueError, match="model_profile"):
        _provider(model_profile="no-such-model")
