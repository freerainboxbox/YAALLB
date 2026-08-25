"""Tests for the on-startup dflash VRAM impact cache (providers/dflash_cache.py).

The cache is keyed on a deterministic sha256 of the canonical serialization of
the frozen YAALLB config, stored at /tmp/yaallb/<hex>.json, and populated at
startup for every dflash-mlx provider. Tests use a monkeypatched CACHE_DIR so
they never touch the real /tmp/yaallb.
"""

import json
import math
import pathlib

import pytest

from providers import dflash_cache as c


def _provider(alias="m1", model_ref="/tmp/t", draft_ref="/tmp/d"):
    return type(
        "DflashP",
        (),
        {
            "_type_id": "dflash-mlx",
            "_instance_id": 0,
            "alias": alias,
            "model_ref": model_ref,
            "draft_ref": draft_ref,
        },
    )()


def _synthetic_engine(model_ref, draft_ref):
    return (123, 45)


DRAFT_CONFIG = {
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_attention_heads": 4,
    "hidden_size": 64,
    "head_dim": 16,
    "sink_size": 64,
    "window_size": 1024,
    "parallel_tokens": 16,
}


# --------------------------------------------------------------------------- #
# Cache key (sha256 of canonical serialization of the frozen config)
# --------------------------------------------------------------------------- #
def test_cache_key_deterministic():
    cfg = {"vram_limit_mb": 100, "yaallb": {"ctx_length": 4096}, "ds4": []}
    assert c.cache_key(cfg) == c.cache_key(cfg)


def test_cache_key_config_sensitive():
    cfg = {"vram_limit_mb": 100, "yaallb": {"ctx_length": 4096}}
    changed = dict(cfg)
    changed["vram_limit_mb"] = 999
    assert c.cache_key(changed) != c.cache_key(cfg)


def test_cache_key_insensitive_to_key_order():
    a = {"x": 1, "y": {"z": 2}}
    b = {"y": {"z": 2}, "x": 1}
    assert c.cache_key(a) == c.cache_key(b)


def test_cachename_is_hex_json_under_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CACHE_DIR", tmp_path)
    cfg = {"vram_limit_mb": 100}
    name = c.cachename(cfg)
    assert name == f"{c.cache_key(cfg)}.json"
    assert c.cache_path(cfg) == tmp_path / name


# --------------------------------------------------------------------------- #
# Recall / compute round-trip
# --------------------------------------------------------------------------- #
def test_recall_miss_then_compute_then_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CACHE_DIR", tmp_path)
    draft_dir = tmp_path / "d"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))

    providers = [_provider(draft_ref=str(draft_dir))]
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"vram_limit_mb": 100}))

    # First call: no cache file -> compute for every dflash provider.
    cache = c.ensure_dflash_cache(str(cfg_path), providers, _synthetic_engine)
    assert list(cache.keys()) == ["dflash-mlx"]  # future-proof top-level key
    entry = cache["dflash-mlx"]["m1"]
    assert entry["target_weight_bytes"] == 123
    assert entry["draft_weight_bytes"] == 45
    assert entry["draft_kv_bytes"] > 0
    assert entry["draft_context_bytes"] > 0
    assert providers[0]._vram_cache is cache

    # Cache file written at /tmp/yaallb/<hex>.json.
    written = [f for f in tmp_path.iterdir() if f.suffix == ".json" and f != cfg_path]
    assert len(written) == 1
    assert written[0].name == c.cachename({"vram_limit_mb": 100})
    stored = json.loads(written[0].read_text())
    assert stored == cache

    # Second call: cache exists -> recall (found-cache event), no recompute.
    cache2 = c.ensure_dflash_cache(str(cfg_path), providers, _synthetic_engine)
    assert cache2 == cache


def test_recall_returns_none_on_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CACHE_DIR", tmp_path)
    assert c.recall({"vram_limit_mb": 100}) is None


def test_no_dflash_providers_skips_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CACHE_DIR", tmp_path)
    other = type("P", (), {"_type_id": "lms", "_instance_id": 0})()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"vram_limit_mb": 100}))
    cache = c.ensure_dflash_cache(str(cfg_path), [other], _synthetic_engine)
    assert cache == {"dflash-mlx": {}}
    # no cache file written when there are no dflash providers
    assert not (tmp_path / c.cachename({"vram_limit_mb": 100})).exists()


def test_compute_impact_without_draft_zeros(tmp_path):
    draft_dir = tmp_path / "d"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
    impact = c.compute_impact(str(tmp_path / "t"), str(draft_dir), _synthetic_engine)
    assert impact["draft_kv_bytes"] > 0 and impact["draft_context_bytes"] > 0
    impact2 = c.compute_impact(str(tmp_path / "t"), None, _synthetic_engine)
    assert impact2["draft_weight_bytes"] == 45
    assert impact2["draft_kv_bytes"] == 0
    assert impact2["draft_context_bytes"] == 0


# --------------------------------------------------------------------------- #
# Real-MLX engine (skips without mlx): compute engine == metadata A
# --------------------------------------------------------------------------- #
try:
    import mlx
    import mlx_lm  # noqa

    HAS_MLX = True
except Exception:  # pragma: no cover
    HAS_MLX = False


def _metadata_a_bytes(model_dir: pathlib.Path) -> int:
    """Approach A: sum safetensors header shape x dtype itemsize."""
    from safetensors import safe_open

    from providers.dflash_vram import dtype_itemsize

    total = 0
    for shard in model_dir.glob("model*.safetensors"):
        with safe_open(str(shard), framework="np") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                total += math.prod(sl.get_shape()) * dtype_itemsize(str(sl.get_dtype()))
    return total


@pytest.mark.skipif(not HAS_MLX, reason="mlx/mlx_lm not installed")
def test_real_mlx_compute_engine_matches_metadata(tmp_path):
    import json

    import mlx.nn as nn
    from mlx_lm.models import llama

    from providers.dflash_vram import compute_weights

    cfg = {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "intermediate_size": 128,
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
    }

    tdir, ddir = tmp_path / "t", tmp_path / "d"
    tdir.mkdir()
    ddir.mkdir()

    args = llama.ModelArgs.from_dict(cfg)
    target = llama.Model(args)
    target.save_weights(str(tdir / "model.safetensors"))
    (tdir / "config.json").write_text(json.dumps(cfg))

    dcfg = dict(cfg)
    dcfg["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    args = llama.ModelArgs.from_dict(cfg)
    draft = llama.Model(args)
    nn.quantize(draft, group_size=32, bits=4)
    draft.save_weights(str(ddir / "model.safetensors"))
    (ddir / "config.json").write_text(json.dumps(dcfg))

    t_w, d_w = compute_weights(str(tdir), str(ddir))
    assert t_w == _metadata_a_bytes(tdir)  # target via B == metadata A
    assert d_w == _metadata_a_bytes(ddir)  # draft via A == metadata A
    assert d_w < t_w  # quantized draft smaller than unquantized target
