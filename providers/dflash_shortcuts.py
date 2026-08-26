"""DFlash shortcut registry + model-path resolution for VRAM estimation.

Mirrors ``dflash-mlx/dflash_mlx/runtime/registry.py`` (``MODEL_SUPPORT_SPECS``,
``resolve_model_support_spec``, ``resolve_optional_draft_ref``). dflash_mlx is
not importable in YAALLB's process (it is spawned as a subprocess), so the
small, stable registry is duplicated here — keep it in sync with that file.

Resolution searches the local HF Hub cache (``huggingface_hub.scan_cache_dir``)
so a repo-id model_ref/draft_ref that was downloaded via ``snapshot_download``
/ ``huggingface-cli download`` resolves to its snapshot directory for VRAM
estimation. A missing download raises ``FileNotFoundError`` so startup can
print download help and exit; a missing draft_ref for a non-shortcut raises
``DflashDraftRequiredError``.
"""

import os

# (base_name, draft_ref) pairs — copied from dflash-mlx MODEL_SUPPORT_SPECS.
MODEL_SUPPORT_SPECS = (
    ("Qwen3.5-4B", "z-lab/Qwen3.5-4B-DFlash"),
    ("Qwen3.5-9B", "z-lab/Qwen3.5-9B-DFlash"),
    ("Qwen3.5-27B", "z-lab/Qwen3.5-27B-DFlash"),
    ("Qwen3.5-35B-A3B", "z-lab/Qwen3.5-35B-A3B-DFlash"),
    ("Qwen3.6-27B", "z-lab/Qwen3.6-27B-DFlash"),
    ("Qwen3.6-35B-A3B", "z-lab/Qwen3.6-35B-A3B-DFlash"),
    ("Qwen3.8-27B", "z-lab/Qwen3.8-27B-DFlash2"),
    ("Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    ("Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    ("gemma-4-31b-it", "z-lab/gemma-4-31B-it-DFlash"),
    ("gemma-4-26b-a4b-it", "z-lab/gemma-4-26B-A4B-it-DFlash"),
)

DRAFT_REGISTRY: dict[str, str] = {base: draft for base, draft in MODEL_SUPPORT_SPECS}

_NORMALIZED = {base.lower(): (base, draft) for base, draft in MODEL_SUPPORT_SPECS}


class DflashDraftRequiredError(Exception):
    def __init__(self, model_ref: str) -> None:
        super().__init__(
            f"model_ref '{model_ref}' is not a DFlash shortcut and no draft_ref "
            "was specified; add \"draft_ref\" to this provider's config"
        )


def _strip_model_org(model_ref: str) -> str:
    return str(model_ref).rsplit("/", 1)[-1].strip()


def resolve_model_support_spec(model_ref: str):
    """Return the ``(base_name, draft_ref)`` shortcut spec for ``model_ref``.

    Matches like dflash-mlx: strip the org, lowercase, exact match then prefix
    match (e.g. ``mlx-community/Qwen3.8-27B-4bit`` -> base ``Qwen3.8-27B``).
    Returns None when the model is not a DFlash shortcut.
    """
    lowered = _strip_model_org(model_ref).lower()
    exact = _NORMALIZED.get(lowered)
    if exact is not None:
        return exact
    bases = [
        base
        for base in _NORMALIZED
        if lowered == base
        or lowered.startswith(base + "-")
        or lowered.startswith(base + "_")
    ]
    if not bases:
        return None
    return _NORMALIZED[max(bases, key=len)]


def resolve_draft_ref(model_ref: str, draft_ref: str | None) -> str | None:
    """Effective draft ref: the explicit one, else the shortcut default.

    Raises ``DflashDraftRequiredError`` when no draft_ref is given for a
    model_ref that is not part of one of the shortcuts.
    """
    if draft_ref:
        return draft_ref
    spec = resolve_model_support_spec(model_ref)
    if spec is None:
        raise DflashDraftRequiredError(model_ref)
    return spec[1]


def _find_in_hf_cache(repo_id: str, cache_dir=None) -> str | None:
    """Locate a downloaded repo's snapshot dir in the local HF Hub cache.

    ``cache_dir`` may be passed explicitly (tests); when None,
    ``scan_cache_dir`` uses the ``HF_HUB_CACHE``/default cache path.
    """
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return None
    for repo in info.repos:  # frozenset of CachedRepoInfo
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:  # frozenset of CachedRevisionInfo
            snapshot = rev.snapshot_path
            if (snapshot / "config.json").exists():
                return str(snapshot)
    return None


def resolve_model_path(model_ref: str, cache_dir=None) -> str:
    """Resolve ``model_ref``/``draft_ref`` to a real local model directory.

    Prefers an existing local directory; else searches the HF Hub cache for a
    downloaded snapshot; else raises ``FileNotFoundError``. ``cache_dir`` may
    be passed explicitly (tests) — otherwise the HF_HUB_CACHE/default path.
    """
    if os.path.isdir(model_ref):
        return os.path.abspath(model_ref)
    cached = _find_in_hf_cache(model_ref, cache_dir=cache_dir)
    if cached is not None:
        return cached
    raise FileNotFoundError(
        f"model '{model_ref}' not found locally or in the HF Hub cache"
    )


def resolve_refs(model_ref: str, draft_ref: str | None, cache_dir=None):
    """Resolve both model and draft to local paths.

    Returns ``(model_path, draft_path, effective_draft_ref)``. Raises
    ``FileNotFoundError`` for a missing download and ``DflashDraftRequiredError``
    when a non-shortcut model has no draft_ref.
    """
    model_path = resolve_model_path(model_ref, cache_dir=cache_dir)
    eff_draft = resolve_draft_ref(model_ref, draft_ref)
    draft_path = (
        resolve_model_path(eff_draft, cache_dir=cache_dir) if eff_draft else None
    )
    return model_path, draft_path, eff_draft
