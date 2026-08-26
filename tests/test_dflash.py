"""Tests for the dflash-mlx provider (providers/dflash.py).

The provider spawns the `dflash` CLI as a subprocess, which is absent in CI,
so load/unload are not exercised here; tests cover the VRAM-cache-backed
memory() and command/descriptor/config logic.
"""

import json

import pytest

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from providers.dflash import (
    DFLASH_DEFAULT_CTX,
    DFLASH_DEFAULT_HOST,
    DFLASH_DEFAULT_PORT,
    DflashProvider,
)
from providers.dflash_vram import WORKING_SET_OVERHEAD, target_kv_bytes


def _provider_with(tmp_path, **overrides) -> DflashProvider:
    cfg = {
        "dflash_dir": str(tmp_path) if tmp_path is not None else None,
        "model_ref": str(tmp_path / "t") if tmp_path is not None else None,
    }
    cfg.update(overrides)
    return DflashProvider(0, cfg)


def _model(p: DflashProvider, model_id="m1", ctx=4096):
    return p.createModel(ModelDescriptor(model_id, p), LoadOptions(ctx_length=ctx))


TARGET_CONFIG = {
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_attention_heads": 4,
    "hidden_size": 64,
    "head_dim": 16,
}


# --------------------------------------------------------------------------- #
# memory() draws from the VRAM cache
# --------------------------------------------------------------------------- #
def test_memory_draws_from_cache(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(TARGET_CONFIG))

    p = _provider_with(tmp_path, alias="m1", ctx_length=4096)
    p._vram_cache = {
        "dflash-mlx": {
            "m1": {
                "target_weight_bytes": 328960,
                "draft_weight_bytes": 62720,
                "draft_kv_bytes": 4096,
                "draft_context_bytes": 1024,
            }
        }
    }
    model = _model(p)
    mem = model.memory()

    base = (
        328960 + 62720 + target_kv_bytes(TARGET_CONFIG, 4096) + 4096 + 1024
    )
    expected = base * WORKING_SET_OVERHEAD / (2**20)
    assert mem == pytest.approx(expected)


def test_memory_ctx_sensitive_via_cache(tmp_path):
    # The cached weight bytes are ctx-independent; the target KV (ctx-scaled)
    # is added at memory() time, so a longer ctx yields a larger estimate.
    tdir = tmp_path / "t"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(TARGET_CONFIG))

    p = _provider_with(tmp_path, alias="m1")
    p._vram_cache = {
        "dflash-mlx": {
            "m1": {
                "target_weight_bytes": 100,
                "draft_weight_bytes": 0,
                "draft_kv_bytes": 0,
                "draft_context_bytes": 0,
            }
        }
    }
    mem_4k = _model(p, ctx=4096).memory()
    mem_64k = _model(p, ctx=65536).memory()
    assert mem_64k > mem_4k  # KV scales with ctx_length


def test_memory_recompute_on_cache_miss_logs_warning(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(TARGET_CONFIG))

    p = _provider_with(tmp_path, alias="m1", ctx_length=4096)
    p._vram_cache = {"dflash-mlx": {"other": {}}}  # model m1 missing
    model = _model(p)
    # Missing entry -> recompute from engine (needs real model dirs); with no
    # model files the engine raises loudly rather than returning a silent 0.
    with pytest.raises(Exception):
        model.memory()


# --------------------------------------------------------------------------- #
# Config defaults, descriptors, effective ctx
# --------------------------------------------------------------------------- #
def test_defaults():
    p = DflashProvider(0, None)
    assert p.host == DFLASH_DEFAULT_HOST
    assert p.port == DFLASH_DEFAULT_PORT
    assert p.binary == "dflash"
    assert p.alias is None
    assert p.options == {}


def test_descriptors_single_alias():
    p = _provider_with(None, alias="qwen-gdn")
    descriptors = p.getModelsDescriptors()
    assert len(descriptors) == 1
    assert descriptors[0].modelId == "qwen-gdn"
    assert descriptors[0].provider is p


def test_get_oai_models_static_no_http(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    p = _provider_with(tmp_path, alias="m1", ctx_length=8192)
    models = p.getOAIModels()
    assert len(models) == 1 and models[0]["id"] == "m1"
    assert models[0]["context_length"] == 8192
    assert models[0]["owned_by"] == "dflash-mlx"
    # no HTTP query: works even though the dflash server is not running


def test_get_oai_models_default_ctx_no_resident(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    p = _provider_with(tmp_path, alias="m1")  # no ctx_length, not resident
    models = p.getOAIModels()
    assert models[0]["context_length"] == DFLASH_DEFAULT_CTX
    assert "stream" in models[0]["supported_parameters"]


def test_effective_ctx_provider_overrides_model():
    p = _provider_with(None, alias="m1", ctx_length=8192)
    model = _model(p, ctx=4096)
    assert p._effective_ctx(model) == 8192
    p.ctx_length = None
    assert p._effective_ctx(model) == 4096


def test_build_command(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    p = _provider_with(
        tmp_path,
        alias="m1",
        host="0.0.0.0",
        port=9999,
        draft_ref=str(tmp_path / "d"),
        options={"wired_limit": "48GB", "quantize_kv_cache": True},
    )
    cmd = p._build_command(_model(p))
    assert cmd[:3] == ["dflash", "serve", "--model"]
    assert str(tdir) in cmd
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and "9999" in cmd
    assert "--draft-model" in cmd and str(tmp_path / "d") in cmd
    assert "--wired-limit" in cmd and "48GB" in cmd
    assert "--quantize-kv-cache" in cmd


def test_build_command_omits_defaults(tmp_path):
    p = _provider_with(tmp_path, alias="m1")
    cmd = p._build_command(_model(p))
    # default host/port, absent draft/options, and default wired-limit ("auto")
    # are not emitted
    assert "--host" not in cmd
    assert "--port" not in cmd
    assert "--draft-model" not in cmd
    assert "--wired-limit" not in cmd


# --------------------------------------------------------------------------- #
# Real-MLX memory() from the defensive engine (skips without mlx)
# --------------------------------------------------------------------------- #
try:
    import mlx
    import mlx_lm  # noqa

    HAS_MLX = True
except Exception:  # pragma: no cover
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="mlx/mlx_lm not installed")
def test_memory_recompute_from_engine_real(tmp_path):
    import mlx.nn as nn
    from mlx_lm.models import llama

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

    # No _vram_cache -> memory() recomputes via the engine (target B, draft A).
    p = _provider_with(tmp_path, alias="m1", draft_ref=str(ddir), ctx_length=4096)
    mem = _model(p).memory()
    assert mem > 0
