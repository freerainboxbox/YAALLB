"""On-startup cache of pre-computed VRAM impact components for providers that
cannot import their model classes in-process (dflash-mlx is the first).

Convention (see README "Cached VRAM estimates"):
- Cache key: sha256 of the canonical serialization of the frozen YAALLB
  config (``frozendict.deepfreeze`` -> canonical JSON bytes). This is
  deterministic across processes — ``hash()`` of a frozendict salts str
  hashes per-process (PYTHONHASHSEED), so it could never be recalled across
  restarts — and it changes whenever config.json changes (a config change
  yields a different key -> a different cache file, so no stale recall).
  ``sort_keys=True`` makes the key insensitive to key order in config.json.
- Cache file: ``/tmp/yaallb/<sha256 hex>.json``.
- Content: ``{"<provider_type>": {model_id: {ctx-independent components}}}``.
  The components (weight bytes + fixed analytical terms) are ctx-independent
  so ``Model.memory()`` can add the ctx-scaled target KV at call time — a
  single flat projected-MiB in the cache would go stale for long-context
  requests. ``compute_weights`` is injected so tests pass the pure-Python
  synthetic engine and production wires the real mlx_lm/A engine.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import log
from frozendict import deepfreeze

from providers.dflash_shortcuts import (
    DflashDraftRequiredError,
    MODEL_SUPPORT_SPECS,
    resolve_draft_ref,
    resolve_model_path,
    resolve_model_support_spec,
)
from providers.dflash_vram import draft_context_bytes, draft_kv_bytes

# Shared cache directory for cached-VRAM-estimate providers.
CACHE_DIR = Path("/tmp/yaallb")


def freeze_config(config: dict):
    """Recursively freeze the config: dicts -> frozendict, lists -> tuples."""
    return deepfreeze(config)


def _canonical_json_bytes(config: dict) -> bytes:
    """Stable, cross-process serialization of the frozen config."""
    frozen = freeze_config(config)
    return json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def cache_key(config: dict) -> str:
    """Deterministic cache key: sha256 of the canonical frozen-config bytes."""
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def cachename(config: dict) -> str:
    """The cache filename ``<hex_string_hash>.json`` for this config."""
    return f"{cache_key(config)}.json"


def cache_path(config: dict) -> Path:
    return CACHE_DIR / cachename(config)


def recall(config: dict) -> dict | None:
    """Load the cache for this config if it exists; log the found-cache event.

    Returns the cache dict (``{"dflash-mlx": {model_id: components}}``) or
    None when the cache file does not exist yet.
    """
    path = cache_path(config)
    if not path.exists():
        return None
    log.info(f"dflash VRAM cache found: {path}")
    with open(path) as f:
        return json.load(f)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_impact(
    model_path: str, draft_path: str | None, compute_weights
) -> dict:
    """ctx-independent footprint components for one dflash provider.

    ``compute_weights(model_path, draft_path)`` returns
    ``(target_weight_bytes, draft_weight_bytes)``; it is the injected engine
    (pure-Python synthetic in tests, real mlx_lm/A in production). The draft
    KV and block-diffusion context terms are analytical from the draft's
    config.json and do not scale with ctx_length. Both paths are already
    resolved local model directories (see dflash_shortcuts.resolve_refs).
    """
    target_weight_bytes, draft_weight_bytes = compute_weights(
        model_path, draft_path
    )
    impact = {
        "target_weight_bytes": target_weight_bytes,
        "draft_weight_bytes": draft_weight_bytes,
    }
    if draft_path is not None:
        draft_config = _read_json(os.path.join(draft_path, "config.json"))
        impact["draft_kv_bytes"] = draft_kv_bytes(draft_config)
        impact["draft_context_bytes"] = draft_context_bytes(draft_config)
    else:
        impact["draft_kv_bytes"] = 0
        impact["draft_context_bytes"] = 0
    return impact


def _default_compute_weights(model_path: str, draft_path: str | None):
    # Production engine: target via mlx_lm lazy graph (Approach B), draft via
    # safetensors metadata (Approach A — dflash classes aren't importable
    # in-process). Imported lazily so the cache module stays importable in
    # environments without mlx (e.g. pure-Python tests of the cache logic).
    from providers.dflash_vram import compute_weights

    return compute_weights(model_path, draft_path)


# --------------------------------------------------------------------------- #
# Helpful download messages (shown on a missing model, then exit non-zero)
# --------------------------------------------------------------------------- #
def _print_download_command(repo_id: str) -> None:
    print(f"  hf download {repo_id}", file=sys.stderr)
    print(
        f"  python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{repo_id}')\"",
        file=sys.stderr,
    )


def _print_download_help(model_ref: str) -> None:
    """Print download commands for a missing model_ref (and its default draft)."""
    spec = resolve_model_support_spec(model_ref)
    print(
        f"ERROR: dflash-mlx model_ref '{model_ref}' is not downloaded locally "
        "(not found on disk or in the HF Hub cache).",
        file=sys.stderr,
    )
    print("Download the target model with a huggingface_hub command:", file=sys.stderr)
    _print_download_command(model_ref)
    if spec is not None:
        print(
            f"'{model_ref}' is a DFlash shortcut "
            f"(base: {spec[0]} -> drafter: {spec[1]}); "
            "its default drafter must also be downloaded:",
            file=sys.stderr,
        )
        _print_download_command(spec[1])
    else:
        print(
            "If you meant a DFlash shortcut, use the base name (e.g. "
            f"'Qwen3.8-27B'); shortcuts: "
            + ", ".join(base for base, _ in MODEL_SUPPORT_SPECS),
            file=sys.stderr,
        )


def _print_draft_download_help(draft_ref: str) -> None:
    print(
        f"ERROR: dflash-mlx draft_ref '{draft_ref}' is not downloaded locally "
        "(not found on disk or in the HF Hub cache).",
        file=sys.stderr,
    )
    print("Download the drafter with a huggingface_hub command:", file=sys.stderr)
    _print_download_command(draft_ref)


def _resolve_provider_paths(provider) -> None:
    """Resolve model/draft to local paths, storing them on the provider.

    On a missing download (or a non-shortcut model with no draft_ref) this
    prints the helpful message and exits with a non-zero code — no raw error
    output is shown.
    """
    model_ref = provider.model_ref
    try:
        model_path = resolve_model_path(model_ref)
    except FileNotFoundError:
        _print_download_help(model_ref)
        sys.exit(1)
    try:
        eff_draft = resolve_draft_ref(model_ref, getattr(provider, "draft_ref", None))
    except DflashDraftRequiredError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    draft_path = None
    if eff_draft:
        try:
            draft_path = resolve_model_path(eff_draft)
        except FileNotFoundError:
            _print_draft_download_help(eff_draft)
            sys.exit(1)
    provider.model_path = model_path
    provider.draft_path = draft_path
    provider.draft_ref_eff = eff_draft


def ensure_dflash_cache(
    config_path: str, providers, compute_weights=None
) -> dict:
    """Recall-or-compute the dflash VRAM cache at startup.

    Reads config.json, derives the deterministic cache key, and either loads
    the existing cache file (logging the found-cache event) or computes the
    VRAM impact for every dflash-mlx provider, logs progress per provider,
    and stores ``{"dflash-mlx": {model_id: components}}`` at
    ``/tmp/yaallb/<hex>.json``. The cache is attached to each dflash provider
    (``provider._vram_cache``) so ``Model.memory()`` can draw from it.

    Model/draft paths are resolved first (local dir or HF Hub cache, shortcut
    draft auto-picked); a missing download or a non-shortcut model without a
    draft_ref prints a helpful message and exits non-zero.
    """
    with open(config_path) as f:
        config = json.load(f)

    dflash_providers = [p for p in providers if p._type_id == "dflash-mlx"]
    for provider in dflash_providers:
        _resolve_provider_paths(provider)

    cached = recall(config)
    if cached is not None:
        _attach(cached, providers)
        return cached

    if not dflash_providers:
        log.info("no dflash providers configured; skipping VRAM cache build")
        return {"dflash-mlx": {}}

    if compute_weights is None:
        compute_weights = _default_compute_weights

    cache = {"dflash-mlx": {}}
    for provider in dflash_providers:
        model_id = getattr(provider, "alias", None)
        if model_id is None:
            log.warning(
                f"dflash provider #{getattr(provider, '_instance_id', 0)} "
                "has no alias; skipping cache entry"
            )
            continue
        log.info(
            f"precomputing dflash VRAM impact model={model_id} "
            f"provider={provider._type_id}#{getattr(provider, '_instance_id', 0)}"
        )
        impact = compute_impact(
            provider.model_path, provider.draft_path, compute_weights
        )
        cache["dflash-mlx"][model_id] = impact

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(config)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"dflash VRAM cache computed and stored: {path}")

    _attach(cache, providers)
    return cache


def _attach(cache: dict, providers) -> None:
    """Attach the cache to each dflash-mlx provider for ``Model.memory()``."""
    for provider in providers:
        if provider._type_id == "dflash-mlx":
            provider._vram_cache = cache
