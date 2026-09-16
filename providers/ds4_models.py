"""Which model IDs a ds4 tree serves, per model family.

ds4 does not have a model list its caller can read: one `ds4-server` process
serves whatever the GGUF passed at startup *is*, under a fixed set of aliases
(`deepseek-v4-flash`, `qwen3.8-flash-next`, ...), and picking a thinking mode is
part of that alias
set (`deepseek-chat` / `deepseek-reasoner` change whether the model thinks; see
ds4_server.c `model_alias_disables_thinking` and
`model_alias_enables_thinking`). YAALLB schedules and routes by model ID, so it
mirrors that table here — the same way `providers/dflash_shortcuts.py` mirrors
dflash-mlx's registry — and confirms the family it is looking at with the ds4
build itself (`model_family` in the estimator output, see
providers/dwarfstar_estimate.py).

Aliases come from ds4_server.c `server_model_alias_known()` /
`server_model_id_from_engine()` / `send_models()`. The first alias of a family is
its primary one, i.e. what `server_model_id_from_engine()` answers. Registering
the thinking aliases at all is what makes thinking mode selectable through
YAALLB: ds4 honours them on the chat endpoint and does not list them in
/v1/models.

They are *routable IDs*, and every one of them names the same resident model:
`ds4-server` answers for the loaded GGUF whichever alias is asked (ds4 uses the
alias only to pick defaults), and the provider is single-resident.
"""

from dataclasses import dataclass


# ds4-server's own default for `-n/--tokens`, which is also what it prints as
# `top_provider.max_completion_tokens` in /v1/models (ds4_server.c parse_options
# default_tokens, append_model_json_values).
DS4_DEFAULT_MAX_COMPLETION_TOKENS = 393216

# ds4's /v1/models reports the same supported_parameters for every model it can
# serve (append_model_json_values), so there is one list here rather than one
# per family.
DS4_SUPPORTED_PARAMETERS = [
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "ignore_eos",
    "stop",
    "seed",
    "stream",
    "reasoning_effort",
]


@dataclass(frozen=True)
class Ds4ModelProfile:
    """One ds4 model family: its IDs, and the context its shape was built for.

    `family` is the `model_family` string a ds4 build reports (tools/
    ds4_estimate.c), so config and auto-detection share one key. `aliases` is
    ordered: primary first, then the extra aliases.

    `native_ctx` is the context length that family's shape is documented to
    have, and only a fallback for when neither the request nor the config asks
    for a context. It is None where ds4 documents no ceiling — YAALLB then keeps
    the context it has always used rather than inventing one.
    """

    family: str
    display_name: str
    aliases: tuple[str, ...]
    native_ctx: int | None = None

    @property
    def primary_alias(self) -> str:
        return self.aliases[0]


DEEPSEEK_V4 = Ds4ModelProfile(
    family="deepseek4",
    # ds4's own shape name; the estimator's model_name replaces it once a GGUF
    # has been opened, which is what tells Flash from PRO or Vision Experimental.
    display_name="DeepSeek V4 Flash",
    # deepseek-chat / deepseek-reasoner are the DeepSeek-compatible aliases ds4
    # uses to force thinking off/on. They are not in its /v1/models list, but
    # chat requests honour them.
    aliases=(
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    # What YAALLB assumed before there was a registry at all: DeepSeek V4's
    # maximum context.
    native_ctx=1000000,
)


QWEN38_FLASH_NEXT = Ds4ModelProfile(
    family="qwen4exp",
    display_name="Qwen3.8 Flash Next",
    # ds4_server.c server_model_alias_known(): three aliases ds4 lists, two
    # no-thinking spellings, and three vendor-prefixed ones. There is no
    # qwen/-prefixed -no-think alias, so none is invented here.
    aliases=(
        "qwen3.8-flash-next",
        "qwen3.8-flash-next-chat",
        "qwen3.8-flash-next-reasoner",
        "qwen3.8-flash-next-no-think",
        "qwen3.8-flash-next-nothink",
        "qwen/qwen3.8-flash-next",
        "qwen/qwen3.8-flash-next-chat",
        "qwen/qwen3.8-flash-next-reasoner",
    ),
    # docs/QWEN38_FLASH_NEXT.md: the native context is 262144; longer needs
    # static YaRN through DS4_QWEN4_YARN_FACTOR, a ds4 environment knob YAALLB
    # has no business assuming on its own.
    native_ctx=262144,
)


DS4_MODEL_PROFILES: tuple[Ds4ModelProfile, ...] = (DEEPSEEK_V4, QWEN38_FLASH_NEXT)

# A ds4 tree whose shape has never been confirmed (no estimator output) has
# always been treated as DeepSeek V4 Flash/PRO, and still is.
DS4_DEFAULT_PROFILE = DEEPSEEK_V4

_BY_FAMILY = {profile.family: profile for profile in DS4_MODEL_PROFILES}


def profile_for(model_family: str | None) -> Ds4ModelProfile | None:
    """The profile for a `model_family` reported by ds4, or None if unknown."""
    return _BY_FAMILY.get(model_family) if model_family else None


def profile_named(name: str | None) -> Ds4ModelProfile | None:
    """A configured `model_profile`, given as its family key or any of its IDs.

    Accepting an alias too keeps config.json writable in the vocabulary clients
    actually use (`"model_profile": "qwen3.8-flash-next"`).
    """
    if not name:
        return None
    if name in _BY_FAMILY:
        return _BY_FAMILY[name]
    for profile in DS4_MODEL_PROFILES:
        if name in profile.aliases:
            return profile
    return None
