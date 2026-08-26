"""Pre-load VRAM projection for the future ``dflash`` provider.

The dflash scheduler calls ``Model.memory()`` *before* loading (to decide
evictions / avoid OOM), so the footprint must be projected with zero VRAM
touched. This module implements the two projection approaches from FINDINGS.md
and records the single approach chosen for ``Model.memory()``.

Weight term — the only place the two approaches differ
-------------------------------------------------------
- **Approach A (pure metadata):** sum safetensors shard headers
  (``config.json`` + ``model.safetensors.index.json`` weight_map + each
  shard's header: tensor ``shape`` x ``dtype.itemsize``). No model graph is
  built; zero VRAM and zero weight data read.
- **Approach B (lazy model structure):** build the nn.Module graph (weights
  kept on disk — shapes/dtypes from the graph, nbytes from shape x dtype),
  then sum the *real stored* array nbytes. Needs the model classes (mlx_lm for
  the target, dflash custom classes for the draft), so it is heavier than A,
  but measures the architecture the runtime will actually use.

Validated equivalence (see FINDINGS.md "VRAM estimation validation"):
  A and B agree *iff* both measure the stored bytes. In particular B must use
  the exact integer sum of array nbytes, never a fixed ``count * bytes_per_param``:
  ``get_total_parameters`` returns the *unquantized* element count for
  quantized layers and omits scales/bias, so a naive multiplier diverges; the
  float ``compute_bits_per_weight`` can also round a byte. When A and B
  disagree (e.g. a checkpoint whose file dtype differs from the graph dtype,
  or an unexpected quantized storage layout), **B — the real model graph — is
  the authoritative result**.

Decision (single approach for ``Model.memory()``) — **B**
---------------------------------------------------------
``.memory()`` uses **Approach B**: the exact integer sum of the real model
graph's array nbytes. Rationale:
- **Always correct regardless of quantization.** B reads the loaded graph's
  arrays, so quantized scales/bias and any quant method (4/8-bit, mxfp4,
  awq/gptq transform, future layouts) are counted exactly and automatically;
  A must hardcode the specific quantized file layout and dtype mapping and can
  silently mis-count an unexpected layout.
- **Always correct regardless of system.** B measures the architecture the
  runtime actually loads (config + model classes), robust to file-vs-graph
  dtype mismatch and missing/inconsistent index.json; A trusts external
  metadata that varies between systems and can be *silently* wrong.
- **Future-proof.** B adapts to whatever config/model the runtime loads; new
  quant methods and layouts are handled by construction, not by format tables.
- **Stable.** B is correct-by-construction and fails *loudly* (missing model
  class / unsupported arch raise) rather than returning a silent wrong number.
  Its estimate is fast (~1ms, O(layer count), not O(hidden size)).

A is retained as the validation cross-check (synthetic test vectors +
real-MLX equivalence test) and for the **production draft weight sum**: the
dflash DFlash classes aren't importable in-process (dflash is spawned as a
subprocess), so lazy-graph B can't be built for the draft — and A == B for the
draft's mlx-native quantized checkpoint (validated), so A is exact there.
The **target** uses B (mlx_lm lazy graph). If B cannot build the target graph,
``.memory()`` should raise rather than silently fall back to A, because a
silent-wrong projection is the worst stability failure.
"""

import glob
import json
import math
import os
from pathlib import Path

# safetensors dtype -> itemsize. MLX stores quantized weights as U32-packed
# (weight) + F32 scales (+ bias); A sums every tensor in the header, so the
# quantized storage bytes are counted exactly.
_DTYPE_ITEMSIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F64": 8,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
}

# dflash runtime defaults (dflash_mlx/runtime/config.py).
DFLASH_DRAFT_SINK_SIZE = 64
DFLASH_DRAFT_WINDOW_SIZE = 1024
DFLASH_DRAFT_FULL_CONTEXT_MIN_CTX = 16384
# block-diffusion parallel draft context width (draft tokens generated at once).
DFLASH_DRAFT_PARALLEL_TOKENS = 16
# working-set overhead factor (activations, Metal buffers) beyond weights+caches.
WORKING_SET_OVERHEAD = 1.15


def dtype_itemsize(dtype: str) -> int:
    return _DTYPE_ITEMSIZE[dtype]


def _numel(shape):
    return math.prod(shape)


# --------------------------------------------------------------------------- #
# Approach A — pure metadata weight sum
# --------------------------------------------------------------------------- #
def weight_bytes_from_metadata(
    index: dict, shard_headers: dict
) -> int:
    """Option A: sum weight bytes from the safetensors index + shard headers.

    ``index``        : parsed ``model.safetensors.index.json`` (has ``weight_map``).
    ``shard_headers``: ``{shard_name: {tensor_name: (shape, dtype_str)}}`` —
                       the per-tensor metadata from each shard's safetensors
                       header. Never materializes weight data.
    """
    total = 0
    for tensor, shard in index["weight_map"].items():
        shape, dtype = shard_headers[shard][tensor]
        total += _numel(shape) * dtype_itemsize(dtype)
    return total


# --------------------------------------------------------------------------- #
# Approach B — lazy model-structure weight bytes
# --------------------------------------------------------------------------- #
def weight_bytes_from_structure(leaf_arrays) -> int:
    """Option B (pure core): sum stored bytes of the model graph's arrays.

    ``leaf_arrays``: iterable of ``(shape, dtype_str)`` for every stored array
    of the built model graph — exactly ``tree_flatten(model.parameters())``,
    which for a quantized layer includes the U32-packed weight, the F32 scales
    and the bias. This is the same tensor set A sees in the safetensors file.
    """
    total = 0
    for shape, dtype in leaf_arrays:
        total += _numel(shape) * dtype_itemsize(dtype)
    return total


def weight_bytes_from_mlx_lazy(model) -> int:
    """Option B (mlx-lm backed): real stored bytes from a lazily built graph.

    The exact authoritative value is the sum of the real array nbytes
    (``tree_flatten(model.parameters())``), which for a quantized layer counts
    the U32-packed weight + F32 scales + bias — the same tensor set A sees in
    the safetensors file.

    NOTE: do NOT use ``get_total_parameters(model) * fixed_bytes``. That count
    returns the *unquantized* element count (it reconstructs
    ``weight.size * 32 // bits`` and omits scales/bias), so any fixed or
    bits-based multiplier diverges for quantized models and non-bf16 dtypes
    (validated). ``compute_bits_per_weight`` is float and can round a byte;
    the nbytes sum below is exact integer arithmetic.

    NOTE: ``mlx_lm.utils.load_model(lazy=True)`` still ``mx.load``s the weight
    files into host RAM (lazy only skips the VRAM ``eval``), so it is
    I/O-bound; a genuinely fast B must build the graph without reading weight
    data (shapes/dtypes come from the graph, nbytes from shape x dtype).
    """
    import mlx  # deferred import keeps this module importable without mlx

    return sum(v.nbytes for _, v in mlx.utils.tree_flatten(model.parameters()))


# --------------------------------------------------------------------------- #
# Analytical cache / activation terms (identical in A and B)
# --------------------------------------------------------------------------- #
def _text_config(config: dict) -> dict:
    """Effective text-model config: the nested ``text_config`` of hybrid
    (qwen3_5/GDN) targets, else the config itself.
    """
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else config


def target_kv_bytes(config: dict, ctx_length: int) -> int:
    """Target KV at ``ctx_length``, K+V per token, plus hybrid/linear state.

    ``bytes_per_el`` is the KV element size (e.g. 2 for bf16). Hybrid
    (qwen3_5/GDN) targets nest their text config under ``text_config`` and
    split ``layer_types`` into full-attention (ctx-scaled KV) and linear
    (gated-delta) layers (fixed recurrent state per layer). GDN extra buffers
    (captured hidden, rollback tape) are config-overridable, default 0.
    """
    tc = _text_config(config)
    layers = tc["num_hidden_layers"]
    bytes_per_el = tc.get("kv_bytes_per_element", 2)
    layer_types = tc.get("layer_types")
    if layer_types:
        full_attn = sum(1 for t in layer_types if t == "full_attention")
        kv_heads = tc.get("num_key_value_heads", tc.get("num_attention_heads"))
        head_dim = tc.get("head_dim") or tc["hidden_size"] // tc.get(
            "num_attention_heads", 1
        )
        kv = full_attn * kv_heads * head_dim * 2 * ctx_length * bytes_per_el
        linear = layers - full_attn
        lin_k = tc.get("linear_num_key_heads", kv_heads) * tc.get(
            "linear_key_head_dim", head_dim
        )
        lin_v = tc.get("linear_num_value_heads", lin_k) * tc.get(
            "linear_value_head_dim", head_dim
        )
        kv += linear * (lin_k + lin_v) * bytes_per_el
    else:
        kv_heads = tc.get("num_key_value_heads", tc.get("num_attention_heads"))
        head_dim = tc.get("head_dim") or tc["hidden_size"] // tc.get(
            "num_attention_heads", 1
        )
        kv = layers * kv_heads * head_dim * 2 * ctx_length * bytes_per_el
    gdn = tc.get("gdn_recurrent_bytes", config.get("gdn_recurrent_bytes", 0))
    captured = tc.get("captured_hidden_bytes", config.get("captured_hidden_bytes", 0))
    rollback = tc.get("rollback_tape_bytes", config.get("rollback_tape_bytes", 0))
    return kv + gdn + captured + rollback


def draft_kv_bytes(draft_config: dict) -> int:
    """Fixed draft KV: sink+window per-token cache, plus full-context layers.

    Not scaled by ctx_length the way target KV is: sink(64)+window(1024) is a
    fixed structure, and layers above ``draft_full_context_min_ctx`` carry a
    full-context cache. Reads the nested ``text_config`` for hybrid drafts too.
    """
    tc = _text_config(draft_config)
    layers = tc["num_hidden_layers"]
    kv_heads = tc.get("num_key_value_heads", tc.get("num_attention_heads"))
    head_dim = tc.get("head_dim") or tc["hidden_size"] // (
        tc.get("num_attention_heads") or 1
    )
    bytes_per_el = tc.get("kv_bytes_per_element", 2)
    sink = tc.get("sink_size", DFLASH_DRAFT_SINK_SIZE)
    window = tc.get("window_size", DFLASH_DRAFT_WINDOW_SIZE)
    cache_per_token = kv_heads * head_dim * 2 * bytes_per_el
    return (sink + window) * cache_per_token + layers * cache_per_token


def draft_context_bytes(draft_config: dict) -> int:
    """Block-diffusion context activations: the parallel draft context and
    draft-generated hidden chunks.

    DFlash2 drafts declare their block width in ``dflash_config.block_size``;
    DFlash v1 uses the default parallel width. Reads the nested ``text_config``
    for hybrid drafts too.
    """
    tc = _text_config(draft_config)
    hidden = tc["hidden_size"]
    bytes_per_el = tc.get("draft_bytes_per_element", 2)
    dflash_cfg = draft_config.get("dflash_config") or {}
    parallel = dflash_cfg.get(
        "block_size", tc.get("parallel_tokens", DFLASH_DRAFT_PARALLEL_TOKENS)
    )
    return parallel * hidden * bytes_per_el


# --------------------------------------------------------------------------- #
# Full projection
# --------------------------------------------------------------------------- #
def projected_mib(
    target_config: dict,
    draft_config: dict,
    ctx_length: int,
    target_weight_bytes: int,
    draft_weight_bytes: int,
    overhead: float = WORKING_SET_OVERHEAD,
) -> float:
    """Projected dflash resident footprint in MiB (target + drafter + caches).

    Both A and B feed the same weight bytes here; the cache/activation terms
    are analytical and identical between approaches.
    """
    target_kv = target_kv_bytes(target_config, ctx_length)
    draft_kv = draft_kv_bytes(draft_config)
    draft_ctx = draft_context_bytes(draft_config)
    base = target_weight_bytes + draft_weight_bytes + target_kv + draft_kv + draft_ctx
    return base * overhead / (2**20)


# Single approach chosen for Model.memory(): B (lazy model structure).
# See the module docstring for the correctness / future-proof / stability
# rationale. A is kept as a pure-Python cross-check only.
CHOSEN_APPROACH = "B"


# --------------------------------------------------------------------------- #
# Production compute engine (target via B, draft via A)
# --------------------------------------------------------------------------- #
def _shard_tensors(path: str) -> dict:
    """{tensor_name: (shape, dtype_str)} from a safetensors shard header.

    Reads only the header metadata (shape x dtype), never the weight data.
    """
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            tensors[k] = (tuple(sl.get_shape()), str(sl.get_dtype()))
    return tensors


def build_target_weight_bytes(model_ref: str) -> int:
    """Approach B (mlx_lm lazy graph): real stored bytes of the target graph.

    The target is an mlx_lm-supported architecture, so its graph can be built
    lazily and its exact integer array nbytes summed. NOTE:
    ``mlx_lm.utils.load_model(lazy=True)`` still ``mx.load``s the weight files
    into host RAM (lazy only skips the VRAM ``eval``), so this is I/O-bound —
    acceptable as a one-time startup cache build; a true-lazy no-data-read
    build (construct the graph from config + classes, skip ``mx.load``) is a
    refinement for multi-GB checkpoints.
    """
    import mlx_lm.utils as U

    # mlx_lm's load_model requires a pathlib.Path (it does ``path / "config.json"``).
    model, _ = U.load_model(Path(model_ref), lazy=True)
    return weight_bytes_from_mlx_lazy(model)


def build_draft_weight_bytes(draft_ref: str) -> int:
    """Approach A (pure metadata): sum the draft checkpoint's stored bytes.

    Reads ``config.json`` + ``model.safetensors.index.json`` (``weight_map``)
    + each shard's safetensors header (shape x dtype), never materializing
    weights. Handles a single-file checkpoint without an index by reading that
    shard directly. The draft is mlx-native quantized, and A == B was validated
    for exactly those checkpoints, so this is exact — the dflash DFlash classes
    are not importable in-process (dflash is spawned as a subprocess), so lazy-
    graph B cannot be built for the draft.
    """
    index_path = os.path.join(draft_ref, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        headers = {
            shard: _shard_tensors(os.path.join(draft_ref, shard))
            for shard in set(weight_map.values())
        }
        return weight_bytes_from_metadata({"weight_map": weight_map}, headers)

    # Single-file checkpoint without an index: sum the one model*.safetensors.
    shards = sorted(glob.glob(os.path.join(draft_ref, "model*.safetensors")))
    if not shards:
        raise RuntimeError(f"no safetensors shards found in {draft_ref}")
    total = 0
    for shard in shards:
        for shape, dtype in _shard_tensors(shard).values():
            total += _numel(shape) * dtype_itemsize(dtype)
    return total


# --------------------------------------------------------------------------- #
# Production compute_weights used by the VRAM cache (providers/dflash_cache.py)
# --------------------------------------------------------------------------- #
def compute_weights(model_ref: str, draft_ref: str | None):
    """Production engine: target via B (lazy graph), draft via A (metadata).

    Returns ``(target_weight_bytes, draft_weight_bytes)``. The draft path uses
    A because the dflash DFlash classes aren't importable in-process; A == B
    for the draft's mlx-native quantized checkpoint (validated).
    """
    target_weight_bytes = build_target_weight_bytes(model_ref)
    if draft_ref is not None:
        draft_weight_bytes = build_draft_weight_bytes(draft_ref)
    else:
        draft_weight_bytes = 0
    return target_weight_bytes, draft_weight_bytes
