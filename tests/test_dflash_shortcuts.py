"""Tests for the DFlash shortcut registry + model-path resolution
(providers/dflash_shortcuts.py).

Mirrors dflash-mlx's registry.py: repo-id model_refs like
``mlx-community/Qwen3.8-27B-4bit`` resolve to the ``Qwen3.8-27B`` shortcut and
its default drafter; missing downloads raise FileNotFoundError; a non-shortcut
model without a draft_ref raises DflashDraftRequiredError.
"""

import pytest

from providers import dflash_shortcuts as s


# --------------------------------------------------------------------------- #
# Shortcut matching (base -> drafter)
# --------------------------------------------------------------------------- #
def test_shortcut_exact():
    spec = s.resolve_model_support_spec("Qwen3.8-27B")
    assert spec == ("Qwen3.8-27B", "z-lab/Qwen3.8-27B-DFlash2")


def test_shortcut_prefix_with_org_and_quant():
    spec = s.resolve_model_support_spec("mlx-community/Qwen3.8-27B-4bit")
    assert spec == ("Qwen3.8-27B", "z-lab/Qwen3.8-27B-DFlash2")


def test_non_shortcut_returns_none():
    assert s.resolve_model_support_spec("mlx-community/SomeModel") is None
    assert s.resolve_model_support_spec("SomeModel") is None


def test_draft_registry_has_known_pairs():
    assert s.DRAFT_REGISTRY["Qwen3.5-27B"] == "z-lab/Qwen3.5-27B-DFlash"
    assert s.DRAFT_REGISTRY["gemma-4-31b-it"] == "z-lab/gemma-4-31B-it-DFlash"


# --------------------------------------------------------------------------- #
# Effective draft ref
# --------------------------------------------------------------------------- #
def test_resolve_draft_ref_explicit_wins():
    assert s.resolve_draft_ref("Qwen3.8-27B", "z-lab/custom") == "z-lab/custom"


def test_resolve_draft_ref_shortcut_default():
    assert (
        s.resolve_draft_ref("mlx-community/Qwen3.8-27B-4bit", None)
        == "z-lab/Qwen3.8-27B-DFlash2"
    )


def test_resolve_draft_ref_non_shortcut_requires():
    with pytest.raises(s.DflashDraftRequiredError):
        s.resolve_draft_ref("mlx-community/SomeModel", None)


# --------------------------------------------------------------------------- #
# Path resolution (local dir / HF cache / missing)
# --------------------------------------------------------------------------- #
def test_resolve_model_path_local_dir(tmp_path):
    assert s.resolve_model_path(str(tmp_path)) == str(tmp_path)


def test_resolve_model_path_missing_raises():
    # a nonexistent local path is never a dir nor a cached HF repo
    with pytest.raises(FileNotFoundError):
        s.resolve_model_path("/nonexistent/yaallb-test-dir")


def _fab_cache(tmp_path, repo_id, snap_id="abc"):
    """Fabricate a HF Hub cache dir and return (cache_dir, snapshot_path)."""
    repo = tmp_path / "hf" / f"models--{repo_id.replace('/', '--')}"
    snap = repo / "snapshots" / snap_id
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    return tmp_path / "hf", snap


def test_find_in_hf_cache(tmp_path):
    cache_dir, snap = _fab_cache(tmp_path, "mlx-community/TestModel", "a1b2c3d4")
    assert s._find_in_hf_cache("mlx-community/TestModel", cache_dir=cache_dir) == str(snap)
    # a repo not in the cache returns None
    assert s._find_in_hf_cache("mlx-community/NotThere", cache_dir=cache_dir) is None


def test_resolve_model_path_hf_cache(tmp_path):
    cache_dir, snap = _fab_cache(tmp_path, "mlx-community/Qwen3.8-27B-4bit")
    assert s.resolve_model_path("mlx-community/Qwen3.8-27B-4bit", cache_dir=cache_dir) == str(snap)


def test_resolve_refs_shortcut_draft(tmp_path):
    # target downloaded into the HF cache; shortcut draft auto-picked and
    # also resolved from the cache
    cache_dir, target_snap = _fab_cache(tmp_path, "mlx-community/Qwen3.8-27B-4bit")
    _, draft_snap = _fab_cache(tmp_path, "z-lab/Qwen3.8-27B-DFlash2")

    model_path, draft_path, eff = s.resolve_refs(
        "mlx-community/Qwen3.8-27B-4bit", None, cache_dir=cache_dir
    )
    assert model_path == str(target_snap)
    assert eff == "z-lab/Qwen3.8-27B-DFlash2"
    assert draft_path == str(draft_snap)
