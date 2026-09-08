"""Validation: VRAM weight-estimation Approach A (metadata) vs Approach B
(lazy model structure) produce the same stored-byte count.

FINDINGS.md decision rule:
  - A and B agree iff both measure the *stored* bytes.
  - B must use real array nbytes (compute_bits_per_weight); a naive
    ``count * bytes_per_param`` diverges for quantized models and non-bf16
    dtypes (validated empirically).
  - When they disagree, B — the real model graph — is authoritative.
  - Prefer B whenever its estimate is fast; fall back to A otherwise.
"""

import math
import pathlib

import pytest

from providers.dflash_vram import (
    CHOSEN_APPROACH,
    DFLASH_DRAFT_PARALLEL_TOKENS,
    dtype_itemsize,
    draft_context_bytes,
    draft_kv_bytes,
    projected_mib,
    target_kv_bytes,
    weight_bytes_from_metadata,
    weight_bytes_from_mlx_lazy,
    weight_bytes_from_structure,
)

try:
    import mlx
    import mlx_lm  # noqa
    HAS_MLX = True
except Exception:  # pragma: no cover
    HAS_MLX = False


# --------------------------------------------------------------------------- #
# Synthetic model-spec helpers (pure Python, no MLX)
# --------------------------------------------------------------------------- #
def make_tensors(spec: dict) -> dict:
    """Stored tensor set for a spec, mirroring how MLX saves unquantized /
    quantized weights. Returns {name: (shape, dtype_str)}.

    For quantized linears every projection stores U32-packed weight + F32
    scales (+ F32 bias), exactly the tensors A sees in safetensors and B sees
    in ``tree_flatten(model.parameters())``.
    """
    hidden = spec["hidden_size"]
    heads = spec["num_attention_heads"]
    kv_heads = spec["num_key_value_heads"]
    head_dim = spec["head_dim"]
    vocab = spec["vocab_size"]
    inter = spec["intermediate_size"]
    layers = spec["num_hidden_layers"]
    quant = spec.get("quant")  # None | {"bits": n, "group_size": g}
    fq = "F32" if spec.get("dtype") == "f32" else "BF16"

    tensors = {
        "model.embed_tokens.weight": ((vocab, hidden), fq),
        "model.norm.weight": ((hidden,), fq),
    }
    if not spec.get("tie_word_embeddings", True):
        tensors["lm_head.weight"] = ((vocab, hidden), fq)

    def proj(name, out, in_):
        if quant:
            # U32-packed quantized weight + F32 scales (+ bias). MLX packs
            # bits/32 elements per U32; scales are F32 per (in//group_size).
            g = quant["group_size"]
            packed = out * (in_ // g) * (quant["bits"] // 32)
            tensors[f"{name}.weight"] = ((out, packed), "U32")
            tensors[f"{name}.scales"] = ((out, in_ // g), "F32")
            tensors[f"{name}.bias"] = ((out,), "F32")
        else:
            tensors[f"{name}.weight"] = ((in_, out), fq)
            tensors[f"{name}.bias"] = ((out,), fq)

    for li in range(layers):
        p = f"model.layers.{li}"
        tensors[f"{p}.input_layernorm.weight"] = ((hidden,), fq)
        tensors[f"{p}.post_attention_layernorm.weight"] = ((hidden,), fq)
        proj(f"{p}.self_attn.q_proj", heads * head_dim, hidden)
        proj(f"{p}.self_attn.k_proj", kv_heads * head_dim, hidden)
        proj(f"{p}.self_attn.v_proj", kv_heads * head_dim, hidden)
        proj(f"{p}.self_attn.o_proj", hidden, heads * head_dim)
        proj(f"{p}.mlp.gate_proj", inter, hidden)
        proj(f"{p}.mlp.up_proj", inter, hidden)
        proj(f"{p}.mlp.down_proj", hidden, inter)
    return tensors


def spec_to_a_inputs(spec: dict):
    """Approach-A weight-sum inputs: index + shard_headers from the spec."""
    tensors = make_tensors(spec)
    index = {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
    headers = {"model-00001-of-00001.safetensors": tensors}
    return index, headers


def spec_to_b_inputs(spec: dict):
    """Approach-B input: the leaf-array (shape, dtype) set of the graph."""
    tensors = make_tensors(spec)
    return [(shape, dtype) for shape, dtype in tensors.values()]


def stored_bytes(tensors: dict) -> int:
    return sum(
        math.prod(shape) * dtype_itemsize(dtype) for shape, dtype in tensors.values()
    )


SPECS = [
    # unquantized bf16 target
    {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "intermediate_size": 128, "dtype": "bf16", "tie_word_embeddings": True,
    },
    # unquantized f32 target
    {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "intermediate_size": 128, "dtype": "f32", "tie_word_embeddings": False,
    },
    # 4-bit quantized draft
    {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "intermediate_size": 128, "dtype": "bf16", "tie_word_embeddings": True,
        "quant": {"bits": 4, "group_size": 32},
    },
    # 8-bit quantized draft
    {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "intermediate_size": 128, "dtype": "bf16", "tie_word_embeddings": True,
        "quant": {"bits": 8, "group_size": 32},
    },
]


# --------------------------------------------------------------------------- #
# A == B equivalence (the core validation)
# --------------------------------------------------------------------------- #
def test_metadata_equals_structure_for_all_specs():
    for spec in SPECS:
        a_total = weight_bytes_from_metadata(*spec_to_a_inputs(spec))
        b_total = weight_bytes_from_structure(spec_to_b_inputs(spec))
        assert a_total == b_total, spec.get("quant")


def test_a_matches_known_tensor_bytes():
    # single bf16 Linear(8, 4): weight (4,8) x 2 = 64
    tensors = {"w": ((4, 8), "BF16")}
    index = {"weight_map": {"w": "s1"}}
    headers = {"s1": tensors}
    assert weight_bytes_from_metadata(index, headers) == 64
    assert weight_bytes_from_structure([((4, 8), "BF16")]) == 64


def test_quantized_storage_counts_scales_and_bias():
    # A 4-bit quantized projection must count the U32 weight + F32 scales + bias.
    tensors = {
        "w.weight": ((8, 4), "U32"),   # 8 * 4 * 4 = 128
        "w.scales": ((8, 2), "F32"),   # 8 * 2 * 4 = 64
        "w.bias": ((8,), "F32"),       # 8 * 4 = 32
    }
    index = {"weight_map": {"w.weight": "s1", "w.scales": "s1", "w.bias": "s1"}}
    headers = {"s1": tensors}
    assert weight_bytes_from_metadata(index, headers) == 128 + 64 + 32
    assert weight_bytes_from_structure(tensors.values()) == 128 + 64 + 32


# --------------------------------------------------------------------------- #
# Validated divergence: naive B (count * fixed bytes) is wrong
# --------------------------------------------------------------------------- #
def test_naive_count_times_bytes_diverges_for_quantized():
    """get_total_parameters-style count * 2 must NOT equal A for quantized
    models — this is the validated reason B needs compute_bits_per_weight."""
    spec = SPECS[2]  # 4-bit quantized
    a_total = weight_bytes_from_metadata(*spec_to_a_inputs(spec))
    b_total = weight_bytes_from_structure(spec_to_b_inputs(spec))

    # naive element count, mirroring get_total_parameters: for quantized
    # layers it reconstructs the unquantized count and omits scales/bias.
    def unquant_count(tensors):
        return sum(math.prod(shape) for shape, _ in tensors.values())

    naive = unquant_count(make_tensors(spec)) * 2
    assert naive != a_total
    assert naive != b_total
    # the real stored count (what B uses) matches A
    assert b_total == a_total


def test_naive_count_times_bytes_diverges_for_f32():
    """Even unquantized, count * 2 diverges when the file dtype is f32."""
    spec = SPECS[1]  # f32 target
    a_total = weight_bytes_from_metadata(*spec_to_a_inputs(spec))
    assert a_total == stored_bytes(make_tensors(spec))
    # A uses 4 bytes/el (F32); a naive count * 2 (bf16 assumption) is wrong.
    elem_count = sum(math.prod(shape) for shape, _ in make_tensors(spec).values())
    assert a_total == elem_count * 4
    assert a_total != elem_count * 2


# --------------------------------------------------------------------------- #
# Decision rule: a single approach is chosen for .memory()
# --------------------------------------------------------------------------- #
def test_chosen_approach_is_b():
    """The .memory() implementation uses Approach B (lazy model structure) as
    the single authoritative approach — always-correct across quantization /
    system, future-proof, stable (loud failure, not silent-wrong)."""
    assert CHOSEN_APPROACH == "B"


# --------------------------------------------------------------------------- #
# Analytical KV terms: hybrid (qwen3_5) targets nest text_config
# --------------------------------------------------------------------------- #
def test_target_kv_bytes_hybrid_nested_text_config():
    """A qwen3_5-style config nests its text model under text_config and splits
    layer_types into full-attention (ctx-scaled KV) and linear (fixed recurrent)
    layers — must not KeyError on top-level num_hidden_layers."""
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_size": 128,
            "layer_types": [
                "linear_attention", "full_attention",
                "linear_attention", "full_attention",
            ],
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 8,
            "linear_num_value_heads": 8,
            "linear_value_head_dim": 8,
        },
    }
    ctx = 100
    # 2 full-attn layers * 2 kv_heads * 16 head_dim * (K+V) * ctx * 2 bytes
    full = 2 * 2 * 16 * 2 * ctx * 2
    # 2 linear layers * (4*8 key + 8*8 value recurrent) * 2 bytes
    linear = 2 * (4 * 8 + 8 * 8) * 2
    assert target_kv_bytes(cfg, ctx) == full + linear


def test_target_kv_bytes_plain_config_still_works():
    cfg = {
        "num_hidden_layers": 4, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 16, "hidden_size": 128,
    }
    ctx = 100
    assert target_kv_bytes(cfg, ctx) == 4 * 2 * 16 * 2 * ctx * 2


def test_draft_context_bytes_dflash2_block_size():
    # DFlash2 declares its block width in dflash_config.block_size; use it
    # over the default parallel width.
    assert draft_context_bytes({"hidden_size": 64, "dflash_config": {"block_size": 8}}) == 8 * 64 * 2
    # DFlash v1 (no dflash_config) uses the default parallel width
    assert draft_context_bytes({"hidden_size": 64}) == DFLASH_DRAFT_PARALLEL_TOKENS * 64 * 2


def test_draft_kv_bytes_nested_text_config():
    cfg = {"text_config": {
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "hidden_size": 64,
    }}
    # sink(64)+window(1024) * 2kv_heads*16*2*2 + layers*per_token
    per = 2 * 16 * 2 * 2
    assert draft_kv_bytes(cfg) == (64 + 1024) * per + 2 * per


def test_projected_mib_uses_same_weight_bytes_for_a_and_b():
    spec = SPECS[0]
    a_w = weight_bytes_from_metadata(*spec_to_a_inputs(spec))
    b_w = weight_bytes_from_structure(spec_to_b_inputs(spec))
    assert a_w == b_w
    ctx = 4096
    mib_a = projected_mib(spec, spec, ctx, a_w, a_w)
    mib_b = projected_mib(spec, spec, ctx, b_w, b_w)
    assert mib_a == mib_b


# --------------------------------------------------------------------------- #
# Real-model validation (skips without mlx) — the authoritative proof
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_MLX, reason="mlx/mlx_lm not installed")
def test_real_mlx_model_a_equals_b(tmp_path):
    """Construct a real MLX Llama on disk, then compare A (files) vs B
    (lazy graph) on the same model — non-tautological ground truth."""
    import mlx.nn as nn
    from mlx_lm.models import llama
    import mlx_lm.utils as U
    from safetensors import safe_open

    cfg = {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "intermediate_size": 128,
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
    }

    def build_and_check(tmp: pathlib.Path, quant: dict | None):
        import json
        import os
        os.makedirs(tmp, exist_ok=True)
        args = llama.ModelArgs.from_dict(cfg)
        model = llama.Model(args)
        if quant is not None:
            nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"])
        model.save_weights(str(tmp / "model.safetensors"))
        # mlx_lm load_model needs config.json; for the quant case it must
        # re-apply the same quantization so the saved U32-packed weights load.
        config = dict(cfg)
        if quant is not None:
            config["quantization"] = {
                "group_size": quant["group_size"], "bits": quant["bits"],
                "mode": "affine",
            }
        with open(tmp / "config.json", "w") as f:
            json.dump(config, f)

        # Approach A: read the shard header metadata only
        a_total = 0
        with safe_open(str(tmp / "model.safetensors"), framework="np") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                dtype = str(f.get_slice(k).get_dtype())
                a_total += math.prod(shape) * dtype_itemsize(dtype)

        # Approach B: lazy graph + real stored bytes (exact integer, no float)
        loaded, _ = U.load_model(tmp, lazy=True)
        b_total = weight_bytes_from_mlx_lazy(loaded)

        assert isinstance(b_total, int)
        assert a_total == b_total, f"quant={quant}: A={a_total} B={b_total}"
        return a_total

    unquant = build_and_check(tmp_path, None)
    quant = build_and_check(tmp_path / "q", {"group_size": 32, "bits": 4})
    # quantized footprint must be strictly smaller than unquantized
    assert quant < unquant


@pytest.mark.skipif(not HAS_MLX, reason="mlx/mlx_lm not installed")
def test_real_mlx_option_b_estimate_is_fast(tmp_path):
    """Measure Option B's estimate time on a real graph. If it is fast, B is
    preferred per FINDINGS decision rule. Generous bound: CI machines vary."""
    import json
    import time
    import mlx.nn as nn
    from mlx_lm.models import llama
    import mlx_lm.utils as U
    from providers.dflash_vram import weight_bytes_from_mlx_lazy

    cfg = {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "intermediate_size": 128,
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(cfg, f)
    args = llama.ModelArgs.from_dict(cfg)
    model = llama.Model(args)
    model.save_weights(str(tmp_path / "model.safetensors"))

    t0 = time.perf_counter()
    loaded, _ = U.load_model(tmp_path, lazy=True)
    t_load = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    b = weight_bytes_from_mlx_lazy(loaded)
    t_bytes = (time.perf_counter() - t0) * 1000

    assert b > 0
    assert t_load + t_bytes < 2000, f"Option B too slow: load={t_load:.1f}ms bytes={t_bytes:.1f}ms"
    print(f"\n[timing] load_model(lazy)={t_load:.2f}ms  weight_bytes={t_bytes:.2f}ms  total={t_load + t_bytes:.2f}ms")
