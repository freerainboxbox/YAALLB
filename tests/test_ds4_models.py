"""ds4's model registry (providers/ds4_models.py) and what a provider serves.

The point is that YAALLB registers the model IDs a ds4 tree really answers to -
including the aliases that turn thinking on and off - and reports them the way
ds4-server itself would, rather than assuming every ds4 tree is DeepSeek V4
Flash/PRO. What ds4 lists is read off the ds4 build itself (`model_aliases` in
the estimator output) whenever it is available, and the registry's own list is
only what is served before a GGUF has been opened; the two are checked against
`tools/ds4_estimate.c` here so neither can drift from the engine in silence.
The aliases a family advertises beyond ds4's list have to be documented in
ds4's own alias tables, so they are asserted literally too: a silent rename
should fail here, not in somebody's config.
"""

import re
from pathlib import Path

import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.routing import lookup_model
from providers.dwarfstar import DwarfStarProvider
from providers.dwarfstar_estimate import DS4_UNKNOWN_MODEL_FAMILY
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


# The ids ds4's chat endpoint honours for one family but keeps out of its own
# /v1/models (ds4_server.c model_alias_disables_thinking /
# model_alias_enables_thinking / server_model_alias_known). They are the only
# thing the registry is allowed to add on top of what ds4 says it serves, and
# there is no such spelling for V4.1, so deepseek41 adds nothing.
DEEPSEEK_V4_THINKING = ["deepseek-chat", "deepseek-reasoner"]
QWEN38_THINKING = [
    "qwen3.8-flash-next-no-think",
    "qwen3.8-flash-next-nothink",
    "qwen/qwen3.8-flash-next",
    "qwen/qwen3.8-flash-next-chat",
    "qwen/qwen3.8-flash-next-reasoner",
]


def _estimator_lists() -> dict:
    """The ids tools/ds4_estimate.c says each family is served under.

    The estimator mirrors ds4_server.c send_models(), so this is the one check
    that the registry's own list of *listed* ids still says what the engine
    says: it reads the C literals instead of another Python copy of them, which
    is the only way a rename upstream can be caught without a built ds4 tree.
    """
    src = (
        Path(__file__).resolve().parent.parent / "tools" / "ds4_estimate.c"
    ).read_text()

    def body(name):
        start = src.index(name)
        return src[start : src.index("\n}\n", start)]

    families = body("static const char *model_family(")
    aliases = body("static void print_model_aliases(")

    arrays = {
        name: tuple(re.findall(r'"([^"]+)"', ids))
        for name, ids in re.findall(
            r"static const char \*const (\w+)\[\] = \{(.*?)\};", aliases, re.S
        )
    }
    # Which array each engine predicate answers with, and which family each
    # predicate is named after; the fallthroughs are the defaults of their
    # functions (both DeepSeek V4).
    by_predicate = dict(
        re.findall(r"if \((ds4_engine_is_\w+)\(e\)\) \{\s*ids = (\w+);", aliases)
    )
    families_by_predicate = dict(
        re.findall(r'if \((ds4_engine_is_\w+)\(e\)\) return "(\w+)";', families)
    )
    default_array = re.search(r"const char \*const \*ids = (\w+);", aliases).group(1)
    # DeepSeek is the default of both functions, but a *checked* one: the
    # estimator only falls through to it for the shapes ds4's own server answers
    # with the deepseek-v4 ids, and the ungated fallback is the unknown-family
    # sentinel. See test_estimator_gates_its_deepseek_default_on_ds4s_shapes.
    default_family = re.search(
        r'return deepseek4_shape_reported\(e\) \? "(\w+)"\s*\n'
        r"\s*: DS4_ESTIMATE_FAMILY_UNKNOWN;",
        families,
    ).group(1)

    served = {
        families_by_predicate[predicate]: arrays[name]
        for predicate, name in by_predicate.items()
    }
    served[default_family] = arrays[default_array]
    assert len(served) == len(DS4_MODEL_PROFILES), served
    return served


# family -> the ids ds4's own build says it serves that family under.
ESTIMATOR_IDS = _estimator_lists()


def test_registry_lists_exactly_what_the_estimator_says_ds4_lists():
    # Nothing here imports ds4: this is the guard against the failure mode the
    # estimator exists to remove, a hand-written id table that silently drifts
    # from the engine it fronts. The registry's listed ids are what is served
    # before a GGUF has been opened, so they have to be the engine's own.
    assert {
        profile.family: profile.aliases for profile in DS4_MODEL_PROFILES
    } == ESTIMATOR_IDS


def test_estimator_gates_its_deepseek_default_on_ds4s_shapes():
    # The estimator's family and alias functions both default to DeepSeek,
    # because ds4's own server does. An *ungated* default is the one silent
    # failure left: a ds4 that grew a fifth shape answers none of the family
    # predicates, would be labelled deepseek4, and would be registered under
    # Flash/PRO ids its GGUF never serves. So the default must stay conditional
    # on ds4's own name for the opened shape, with the sentinel as the fallback.
    src = (
        Path(__file__).resolve().parent.parent / "tools" / "ds4_estimate.c"
    ).read_text()

    shapes = re.search(
        r"static const char \*const deepseek4_shapes\[\] = \{(.*?)\};", src, re.S
    )
    assert shapes, "the estimator lost the DeepSeek shape table"
    assert tuple(re.findall(r'"([^"]+)"', shapes.group(1))) == (
        "DeepSeek V4 Flash",
        "DeepSeek V4 Pro",
    ), "these are the two shapes ds4_server.c answers with the deepseek-v4 ids"

    assert re.search(
        r'return deepseek4_shape_reported\(e\) \? "deepseek4"\s*\n'
        r"\s*: DS4_ESTIMATE_FAMILY_UNKNOWN;",
        src,
    ), "model_family() must not fall through to DeepSeek unchecked"
    # And its aliases must be gated the same way, or the shape would still be
    # advertised under ids it does not answer to.
    assert re.search(
        r'if \(!strcmp\(model_family\(e\), DS4_ESTIMATE_FAMILY_UNKNOWN\)\) \{\s*'
        r"ids = unknown;",
        src,
    )


def test_unknown_family_has_no_registry_profile():
    # What makes the sentinel safe: no profile means the provider keeps the
    # model list config asked for (and says so) instead of presenting DeepSeek's
    # ids for a model it cannot name.
    assert DS4_UNKNOWN_MODEL_FAMILY not in {
        profile.family for profile in DS4_MODEL_PROFILES
    }
    assert profile_for(DS4_UNKNOWN_MODEL_FAMILY) is None


def test_no_registry_alias_is_something_ds4_does_not_honour():
    # Everything the registry adds on top of ds4's list has to be a spelling
    # ds4's chat endpoint accepts, or YAALLB advertises an id whose requests
    # fail. ds4's tables are the only source for these, family by family, and
    # none of them may be one ds4 lists itself (that would list it twice).
    documented = {
        "deepseek4": DEEPSEEK_V4_THINKING,
        "deepseek41": [],
        # ds4 has no qwen/-prefixed no-thinking spelling, so none is invented
        # here either.
        "qwen4exp": QWEN38_THINKING,
        "glm53": [
            "glm-5.3-flash-no-think",
            "glm-5.3-flash-nothink",
            "zai/glm-5.3-flash",
            "zai/glm-5.3-flash-chat",
            "zai/glm-5.3-flash-reasoner",
        ],
        "glm52": [
            "glm-5.2-no-think",
            "glm-5.2-nothink",
            "zai/glm-5.2",
            "zai/glm-5.2-chat",
            "zai/glm-5.2-reasoner",
        ],
    }
    for profile in DS4_MODEL_PROFILES:
        assert profile.thinking_aliases == tuple(documented[profile.family])
        assert not set(profile.thinking_aliases) & set(ESTIMATOR_IDS[profile.family])
        # ... and the family's full table is ds4's list plus exactly those.
        assert profile.served() == (
            *ESTIMATOR_IDS[profile.family], *profile.thinking_aliases
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
        for alias in profile.served():
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
    """A ds4 estimator result, as providers/dwarfstar_estimate.py returns it.

    `model_aliases` follows the family unless given, because a real estimator
    answer always carries the ids of the shape it opened.
    """
    family = overrides.get("model_family", "qwen4exp")
    result = {
        "source": "ds4",
        "model_name": "Qwen3.8 Flash Next",
        "model_family": family,
        "model_aliases": list(ESTIMATOR_IDS.get(family, ESTIMATOR_IDS["deepseek4"])),
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


def test_ds4s_own_id_list_replaces_the_registry_one(no_network, estimator):
    # The registry is a mirror, not an authority: whatever the opened GGUF
    # answers with is what gets routed and budgeted. A ds4 that renames, adds
    # or drops a listed id must not leave YAALLB advertising an id it does not
    # answer (a 400-class failure the client cannot do anything about) or
    # hiding one it does.
    renamed = ["qwen3.8-flash-next-v2", "qwen3.8-flash-next-chat"]
    estimator(_estimate(model_aliases=renamed))
    provider = _provider()
    provider._estimate(8192)

    ids = [d.modelId for d in provider.getModelsDescriptors()]
    assert ids[: len(renamed)] == renamed
    assert "qwen3.8-flash-next" not in ids  # the stale registry id is gone
    # What ds4 does not list but does honour still rides along, because that is
    # how thinking mode stays selectable through YAALLB.
    assert ids[len(renamed) :] == QWEN38_THINKING
    assert [m["id"] for m in provider.getOAIModels()] == ids


def test_a_listed_id_ds4_honours_is_not_listed_twice(no_network, estimator):
    estimator(_estimate(model_aliases=["qwen3.8-flash-next", *QWEN38_THINKING[:2]]))
    provider = _provider()
    provider._estimate(8192)

    ids = [d.modelId for d in provider.getModelsDescriptors()]
    assert len(ids) == len(set(ids))
    assert ids[0] == "qwen3.8-flash-next"


def test_nothing_is_advertised_that_a_ds4_build_does_not_answer(no_network, estimator):
    # The same list, seen through the client-facing listing: /v1/models is what
    # a client like Open WebUI discovers, so every id in it has to be one this
    # tree answers.
    reported = ["qwen3.8-flash-next-v2", "qwen3.8-flash-next-v2-reasoner"]
    estimator(_estimate(model_aliases=reported))
    provider = _provider()
    provider._estimate(8192)

    assert [m["id"] for m in provider.getOAIModels()] == [
        *reported,
        *QWEN38_THINKING,
    ]


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
    assert any("model_profile" in m and "qwen4exp" in m and "although" in m for m in messages)


def test_a_confirmed_profile_is_not_reported_as_a_clash(no_network, estimator, monkeypatch):
    messages = []
    monkeypatch.setattr(
        "providers.dwarfstar.log.info", lambda message: messages.append(message)
    )
    estimator(_estimate())
    provider = _provider(model_profile="qwen4exp")
    provider._estimate(8192)

    reported = [m for m in messages if "model_profile" in m]
    assert len(reported) == 1
    assert "confirmed by ds4" in reported[0] and "although" not in reported[0]


def test_unknown_family_keeps_the_default_and_says_so(no_network, estimator, monkeypatch):
    warnings = []
    monkeypatch.setattr("providers.dwarfstar.log.warning", lambda m: warnings.append(m))
    estimator(_estimate(model_family="some-new-family"))

    provider = _provider()
    provider._estimate(8192)

    assert provider.served_profile is profile_for("deepseek4")
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


# --------------------------------------------------------------------------- #
# custom-prefix: publishing one instance's IDs under a name the operator chose.
#
# ds4's own IDs are fixed, so two trees of one family answer the same strings
# and only the first one in config.json is ever reachable. `custom-prefix` puts
# whatever the operator wants in front of every ID this instance publishes,
# which renames its whole surface at once: descriptors, /v1/models, and with
# them routing, budgeting and eviction.
# --------------------------------------------------------------------------- #


def test_custom_prefix_renames_every_published_id(no_network):
    provider = _provider(**{"custom-prefix": "acme/"})

    assert [d.modelId for d in provider.getModelsDescriptors()] == [
        f"acme/{alias}" for alias in DEEPSEEK_V4_IDS
    ]
    # /v1/models is what a client discovers, so it has to say the routable
    # spelling rather than the one only ds4 knows.
    assert [m["id"] for m in provider.getOAIModels()] == [
        f"acme/{alias}" for alias in DEEPSEEK_V4_IDS
    ]


def test_custom_prefix_is_spelled_with_either_separator(no_network):
    # Every other key of this provider is an underscore name, and `options`
    # already treats a ds4 flag's dashes and underscores as one spelling, so a
    # config written in either vocabulary works.
    dashed = _provider(**{"custom-prefix": "acme/"})
    underscored = _provider(custom_prefix="acme/")

    assert [d.modelId for d in underscored.getModelsDescriptors()] == [
        d.modelId for d in dashed.getModelsDescriptors()
    ]


def test_two_instances_of_one_family_are_routable_apart(no_network):
    # The reason the key exists: without it both instances answer
    # `deepseek-v4-flash` and lookup_model hands every request to whichever
    # instance config.json lists first.
    first = _provider(**{"custom-prefix": "a/", "port": 8000})
    second = _provider(**{"custom-prefix": "b/", "port": 8001})

    # Both trees answer exactly the same ds4 aliases; only the prefix tells
    # them apart from the outside.
    assert first.served_ids == second.served_ids
    assert lookup_model([first, second], "a/deepseek-v4-flash") is first
    assert lookup_model([first, second], "b/deepseek-v4-flash") is second
    # The bare ds4 id belongs to neither of them any more.
    assert lookup_model([first, second], "deepseek-v4-flash") is None


def test_custom_prefix_renames_the_ids_ds4_reports_too(no_network, estimator):
    # The prefix is applied to whatever this instance ends up serving, so it
    # cannot be a list maintained next to the registry the registry replaces.
    renamed = ["qwen3.8-flash-next-v2", "qwen3.8-flash-next-v2-chat"]
    estimator(_estimate(model_aliases=renamed))
    provider = _provider(**{"custom-prefix": "acme/"})
    provider._estimate(8192)

    assert [d.modelId for d in provider.getModelsDescriptors()] == [
        "acme/qwen3.8-flash-next-v2",
        "acme/qwen3.8-flash-next-v2-chat",
        *[f"acme/{alias}" for alias in QWEN38_THINKING],
    ]


def test_a_prefixed_id_reaches_ds4_as_ds4s_own_alias(no_network):
    # ds4 matches model aliases by exact string (ds4_server.c
    # model_alias_disables_thinking / model_alias_enables_thinking), so the
    # prefix has to be gone before the request leaves YAALLB. A
    # `acme/deepseek-chat` forwarded verbatim is an alias ds4 does not know,
    # and thinking would stay on.
    provider = _provider(**{"custom-prefix": "acme/"})

    assert provider.upstream_model_id("acme/deepseek-chat") == "deepseek-chat"
    assert (
        provider.upstream_model_id("acme/qwen3.8-flash-next-nothink")
        == "qwen3.8-flash-next-nothink"
    )
    # An id the prefix is not on is not this instance's to rewrite, and an
    # instance without a prefix never rewrites anything.
    assert provider.upstream_model_id("deepseek-chat") == "deepseek-chat"
    assert _provider().upstream_model_id("deepseek-chat") == "deepseek-chat"


def test_custom_prefix_leaves_ds4s_own_ids_alone(no_network):
    # Everything internal - ds4's alias tables, the registry comparison, the
    # model_overrides a request's own id resolves against - keeps working in
    # the engine's vocabulary; only what is published is renamed.
    provider = _provider(**{"custom-prefix": "acme/"})

    assert provider.served_ids == tuple(DEEPSEEK_V4_IDS)
    assert "acme/deepseek-v4-flash" not in provider.served_ids


def test_conflicting_custom_prefix_spellings_are_rejected(no_network):
    # Both spellings are accepted, so two that disagree are a config nobody
    # can mean, and it says so at startup rather than serving one of them.
    with pytest.raises(ValueError, match="custom-prefix"):
        _provider(**{"custom-prefix": "a/", "custom_prefix": "b/"})


def test_a_non_string_custom_prefix_is_rejected(no_network):
    with pytest.raises(ValueError, match="custom-prefix"):
        _provider(**{"custom-prefix": 42})


def test_an_empty_custom_prefix_publishes_the_engine_ids(no_network):
    # Nothing in front is the same instance with nothing in front.
    provider = _provider(**{"custom-prefix": ""})
    assert [d.modelId for d in provider.getModelsDescriptors()] == DEEPSEEK_V4_IDS
    assert provider.upstream_model_id("deepseek-chat") == "deepseek-chat"
