"""ds4's model registry (providers/ds4_models.py) and what a provider serves.

The point is that YAALLB registers the model IDs a ds4 tree really answers to -
including the aliases that turn thinking on and off - and reports them the way
ds4-server itself would, rather than assuming every ds4 tree is DeepSeek V4
Flash/PRO. The registry itself is a copy of ds4_server.c's alias tables, so the
families that matter are asserted literally: a silent rename should fail here,
not in somebody's config.
"""

import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from providers.dwarfstar import DwarfStarProvider
from providers.ds4_models import (
    DS4_DEFAULT_MAX_COMPLETION_TOKENS,
    DS4_MODEL_PROFILES,
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

# Exactly the ids ds4_server.c server_model_alias_known() accepts for the
# qwen4exp shape: three listed in /v1/models, two no-thinking spellings, and
# three vendor-prefixed ones (ds4 has no qwen/-prefixed -no-think alias).
QWEN38_IDS = [
    "qwen3.8-flash-next",
    "qwen3.8-flash-next-chat",
    "qwen3.8-flash-next-reasoner",
    "qwen3.8-flash-next-no-think",
    "qwen3.8-flash-next-nothink",
    "qwen/qwen3.8-flash-next",
    "qwen/qwen3.8-flash-next-chat",
    "qwen/qwen3.8-flash-next-reasoner",
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


V41_IDS = ["deepseek-v4.1-flash"]
GLM_53_IDS = [
    "glm-5.3-flash",
    "glm-5.3-flash-chat",
    "glm-5.3-flash-reasoner",
    "glm-5.3-flash-no-think",
    "glm-5.3-flash-nothink",
    "zai/glm-5.3-flash",
    "zai/glm-5.3-flash-chat",
    "zai/glm-5.3-flash-reasoner",
]
GLM_52_IDS = [
    "glm-5.2",
    "glm-5.2-chat",
    "glm-5.2-reasoner",
    "glm-5.2-no-think",
    "glm-5.2-nothink",
    "zai/glm-5.2",
    "zai/glm-5.2-chat",
    "zai/glm-5.2-reasoner",
]


def test_the_other_single_machine_families_are_registered(no_network):
    expected = {
        # ds4 gives V4.1 exactly one alias: no thinking variants exist for it.
        "deepseek41": V41_IDS,
        "glm53": GLM_53_IDS,
        "glm52": GLM_52_IDS,
    }
    for family, ids in expected.items():
        provider = _provider(model_profile=family)
        assert [d.modelId for d in provider.getModelsDescriptors()] == ids, family
        assert [m["id"] for m in provider.getOAIModels()] == ids, family
        # None of these families has a documented ceiling, so they keep the
        # context this provider has always used instead of a invented one.
        assert provider._effective_ctx() == 1000000, family


def test_no_alias_is_shared_between_families():
    # Two ds4 instances of different families both answer /v1/models, and
    # routing picks the first provider holding a descriptor: a shared alias
    # would make a Qwen request silently load a GLM tree.
    seen = {}
    for profile in DS4_MODEL_PROFILES:
        for alias in profile.aliases:
            assert alias not in seen, f"{alias} in {seen.get(alias)} and {profile.family}"
            seen[alias] = profile.family


def test_qwen_aliases_are_all_registered(no_network):
    provider = _provider(model_profile="qwen4exp")

    assert [d.modelId for d in provider.getModelsDescriptors()] == QWEN38_IDS
    assert [m["id"] for m in provider.getOAIModels()] == QWEN38_IDS
    # ds4 answers all of them for the loaded GGUF; -chat and -no-think/-nothink
    # reply without thinking, -reasoner insists on it.
    assert {m["name"] for m in provider.getOAIModels()} == {"Qwen3.8 Flash Next"}


def test_qwen_uses_its_own_native_context(no_network):
    # The GGUF decides the model, but --ctx still comes from YAALLB: a Qwen
    # shape sized at 262144 must never inherit DeepSeek's million just because
    # that is what this provider hardcoded before any registry existed.
    provider = _provider(model_profile="qwen3.8-flash-next")
    assert provider._effective_ctx() == 262144
    assert [m["context_length"] for m in provider.getOAIModels()] == [262144] * len(
        QWEN38_IDS
    )

    # A documented ceiling only fills the gap when nothing else asks; the
    # request still wins.
    model = provider.createModel(
        ModelDescriptor("qwen3.8-flash-next", provider), LoadOptions(ctx_length=8192)
    )
    assert provider._effective_ctx(model) == 8192

    # DeepSeek keeps what it always had.
    assert _provider()._effective_ctx() == 1000000
    # ... and a family with no documented ceiling keeps it too, rather than
    # inventing a number ds4 never stated.
    assert profile_for("deepseek4").native_ctx == 1000000


def _estimate(**overrides) -> dict:
    """A ds4 estimator result, as providers/dwarfstar_estimate.py returns it."""
    result = {
        "source": "ds4",
        "model_name": "Qwen3.8 Flash Next",
        "model_family": "qwen4exp",
        "model_id": 5,
        "model_bytes": 1 << 30,
        "support_bytes": 0,
        "vision_bytes": 0,
        "context_bytes": 1 << 20,
        "spec_graph_bytes": 0,
    }
    return {**result, **overrides}


@pytest.fixture
def estimator(monkeypatch):
    """Feeds the provider a canned ds4 estimate, as if the GGUF were opened."""
    import providers.dwarfstar as dsmod

    def install(estimate):
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            return estimate

        monkeypatch.setattr(dsmod, "ds4_estimate", fake)
        return calls

    return install


def test_family_is_detected_from_ds4s_answer(no_network, estimator, monkeypatch):
    estimator(_estimate())
    warnings = []
    monkeypatch.setattr("providers.dwarfstar.log.warning", lambda m: warnings.append(m))

    provider = _provider()
    assert provider.served_profile.family == "deepseek4"  # nothing learned yet

    provider._estimate(8192)

    # ... and now it knows: the aliases switch, and so does the context.
    assert provider.served_profile.family == "qwen4exp"
    assert [d.modelId for d in provider.getModelsDescriptors()][:2] == QWEN38_IDS[:2]
    assert provider._effective_ctx() == 262144
    assert provider._detected_profile is profile_for("qwen4exp")
    assert warnings == []


def test_detection_names_the_model_ds4_says_it_opened(no_network, estimator):
    # Flash and PRO share one profile (and one family); ds4's own shape name is
    # what tells a client which of the two this tree actually holds.
    estimator(_estimate(model_name="DeepSeek V4 Pro", model_family="deepseek4", model_id=1))
    provider = _provider()
    provider._estimate(1000000)

    names = {m["name"] for m in provider.getOAIModels()}
    assert names == {"DeepSeek V4 Pro"}
    assert provider._effective_ctx() == 1000000


def test_configured_profile_wins_over_detection(no_network, estimator, monkeypatch):
    messages = []
    monkeypatch.setattr(
        "providers.dwarfstar.log.info", lambda message: messages.append(message)
    )
    estimator(_estimate())
    provider = _provider(model_profile="deepseek4")
    provider._estimate(8192)

    # An explicit profile is an override, so it is kept - but what ds4 actually
    # reported is still said out loud, because that is the usual shape of "your
    # config and your GGUF disagree".
    assert provider.served_profile.family == "deepseek4"
    assert [d.modelId for d in provider.getModelsDescriptors()] == DEEPSEEK_V4_IDS
    assert any("model_profile" in m and "qwen4exp" in m for m in messages)


def test_unknown_family_keeps_the_default_and_says_so(no_network, estimator, monkeypatch):
    warnings = []
    monkeypatch.setattr("providers.dwarfstar.log.warning", lambda m: warnings.append(m))
    estimator(_estimate(model_family="some-new-family"))

    provider = _provider()
    provider._estimate(8192)

    assert provider.served_profile is profile_for("deepseek4")
    assert [d.modelId for d in provider.getModelsDescriptors()] == DEEPSEEK_V4_IDS
    assert len(warnings) == 1 and "some-new-family" in warnings[0]


def test_a_fallback_estimate_does_not_burn_the_detection_chance(
    no_network, estimator, monkeypatch
):
    warnings = []
    monkeypatch.setattr("providers.dwarfstar.log.warning", lambda m: warnings.append(m))
    # The size-based fallback knows the GGUF's bytes and nothing about its shape.
    estimator(_estimate(model_name=None, model_family=None, model_id=None),)
    provider = _provider()
    provider._estimate(8192)

    assert provider._detected_profile is None
    assert warnings == []
    assert provider._identity_adopted is False


def test_detected_family_also_drives_the_warm_log(no_network, estimator, monkeypatch):
    # warm_dwarfstar_estimates reports the family it priced, so a Qwen tree does
    # not look like a misnamed DeepSeek instance in the startup log.
    from providers.dwarfstar import warm_dwarfstar_estimates

    estimator(_estimate())
    messages = []
    monkeypatch.setattr("providers.dwarfstar.log.info", lambda message: messages.append(message))

    provider = _provider(ctx_length=8192)
    warm_dwarfstar_estimates([provider])

    assert any("qwen4exp" in m for m in messages)


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
