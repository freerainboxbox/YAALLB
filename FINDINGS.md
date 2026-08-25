# Findings: dflash-mlx provider for YAALLB

Explored `dflash-mlx` (Apple Silicon speculative-decoding engine, stock MLX) to
determine how to implement a `dflash` provider in YAALLB. Two items covered:

1. VRAM footprint in MiB when loading a model.
2. Server lifecycle (startup, clean exit).

## 1. VRAM footprint in MiB (PRE-LOAD)

**dflash-mlx has NO in-built pre-load VRAM footprint facility.** Its memory
machinery is entirely *post-load* runtime measurement; nothing computes the
model footprint before weights are resident. The YAALLB scheduler calls
`Model.memory()` **before** loading (to decide evictions / avoid OOM), so the
post-load facilities below are unusable — loading the model to measure its
memory is exactly what we must not do.

### What exists, and why it does not satisfy the pre-load requirement

- `mx.get_peak_memory()` / `mx.get_active_memory()` / `mx.get_cache_memory()`
  (public MLX): measure **after** weights are materialized. Not usable.
- `dflash_mlx.observability.memory.process_memory_snapshot()` /
  `live_memory_payload()`: wraps the above (plus macOS `task_info` and RSS)
  into `mlx_active_bytes`, `mlx_peak_bytes`, `phys_footprint_bytes`, etc.
  Also post-load. Divide by `2**20` for MiB (dflash reports GB/GiB, e.g.
  `_bytes_to_gib` in `server/runtime.py`, `_gb_or_none` in memory.py).
- `dflash_mlx.metal_limits` (`mx.device_info()["max_recommended_working_set_size"]`
  and `memory_size`): gives the device's VRAM **budget** (for the OOM
  comparison), not the model footprint.
- `mlx_lm.utils.get_total_parameters(model)` (public): parameter count from a
  **built** model object's shapes. Requires constructing the nn.Module graph,
  not pure metadata. Only viable pre-load if loaded `lazy=True` (see Option B).

### Pre-load projection must be computed from metadata (pure Python, zero VRAM)

A pre-load estimate must account for the **whole speculative engine** — target
**and drafter** — in the same buckets dflash tracks at runtime (see
`dflash_mlx/engine/memory_waterfall.py`: `target_fa_kv`, `target_gdn_state`,
`rollback_tape`, `draft_kv`, `draft_context_active`, `target_hidden_active`,
`gen_hidden_chunks`, `l1_snapshots`, `l2_disk`, plus `mlx_active` /
`mlx_cache` / `untracked` working-set overhead). A target-only estimate
under-counts badly, because the drafter is a second resident model with its own
KV and activations.

**Drafter footprint — what it is and how to project it.**
The dflash runtime bundle loads **two** models (`dflash_mlx/runtime/bundle.py`
`load_runtime_bundle`): `target_model` (stock MLX) and `draft_model` (custom
DFlash classes via `load_draft_bundle`, default ~1B params, quantized). The
drafter's resident footprint is:

- **Draft weights** — sum draft safetensors shards (Option A) or lazy param
  count × bytes-per-param (Option B); must honor the draft quantization
  (`bits`), which `get_total_parameters` already handles.
- **Draft KV cache** (`draft_kv_bytes`) — `ContextOnlyDraftKVCache` /
  `FullContextDraftKVCache` (`dflash_mlx/model.py`), sized by
  `sink_size` + `window_size` (runtime defaults 64 + 1024;
  `dflash_mlx/runtime/config.py`), plus full-context layers above
  `draft_full_context_min_ctx` (default 16384). This is a **fixed** structure,
  **not scaled by ctx_length** the way target KV is.
- **Block-diffusion context activations** (`draft_context_active_bytes` +
  `gen_hidden_chunks_bytes`) — the 16-token parallel draft context and
draft-generated hidden chunks.
- **Projected full-context** (`forward_projected_context`) — draft hidden
  projected through target layers when above the min-ctx threshold.

So the pre-load projection must be the **sum over both models and all cache
buckets**, not just target weights + a target KV term.

**Option A — pure metadata, no model construction (recommended).**
Read `config.json` + `model.safetensors.index.json` (`weight_map`) + each
shard's safetensors header (tensor `shape` × `dtype.itemsize`) and sum weight
bytes for **both** the target and the draft, without ever materializing weights.
Then add the analytical cache/activation terms from config + runtime defaults:
target KV at `ctx_length` (`num_hidden_layers`, `num_key_value_heads`,
`head_dim`, ×2 for K+V; Qwen-GDN targets carry KV **+ recurrent state +
captured hidden + rollback tape**), the **fixed** draft sink+window KV, and the
block-diffusion context activations; finally a working-set overhead factor. No
model object is built; zero VRAM is touched.

```python
# per tensor: numel = prod(shape); bytes = numel * dtype.itemsize
# target_weights_mib = sum over target shard tensors / 2**20
# draft_weights_mib  = sum over draft shard tensors / 2**20
# target_kv_mib      = layers*num_kv_heads*head_dim*2*ctx*bytes_per_el
#                    + gdn_recurrent + captured_hidden + rollback_tape
# draft_kv_mib       = (sink_size + window_size) * draft_cache_per_token
#                    + full_ctx_layers_above_min_ctx
# draft_ctx_mib      = block_diffusion_parallel * draft_hidden_bytes  # 16-token
# overhead_mib       = working_set (activations, Metal buffers) factor
# projected_mib = target_weights + draft_weights
#               + target_kv + draft_kv + draft_ctx + overhead
```

**Option B — lazy model structure, zero VRAM (MLX-native alternative).**
Build the nn.Module graph (`mlx_lm.utils.load_model(ref, lazy=True)`, and
dflash's `dflash_mlx.runtime.loading.load_draft_bundle` with the custom
`get_draft_model_classes`) and read the **real stored bytes** of the graph
(`sum of array nbytes` over `tree_flatten(model.parameters())`, or
`get_total_parameters × compute_bits_per_weight / 8`), then add KV + overhead
and discard the model. Heavier than A (constructs the graph, needs the custom
DFlash draft classes), but no OOM risk.

⚠️ Two gotchas validated below (see "VRAM estimation validation"):
- `mlx_lm.utils.load_model(lazy=True)` does **not** keep weights on disk — it
  `mx.load`s the full weight files into **host RAM** (lazy only skips the VRAM
  `mx.eval`). So it is I/O-bound (seconds + a multi-GB host-RAM spike for
  production models); a genuinely *fast* B must build the graph without reading
  weight data (shapes/dtypes come from the graph; nbytes from shape × dtype).
- `get_total_parameters` returns the **unquantized** element count
  (`weight.size * 32 // bits`) and omits scales/bias, so `count × fixed_bytes`
  is **wrong** for quantized models and non-bf16 dtypes. The byte multiplier
  must come from the real array nbytes (`compute_bits_per_weight`); prefer the
  exact integer nbytes sum, since the float path can round a byte.

### Decision rule

Compare `projected_mib` against the device budget
(`mx.device_info()["max_recommended_working_set_size"]` — the same budget
dflash's `apply_metal_limits` uses). If `projected_mib + resident_mib >
budget_mib`, evict before loading — never load-then-measure.

**Which weight approach to use.** Option A depends on external metadata
(config/index/shard headers/dtype mapping) matching what the runtime actually
loads; it can diverge from the real model when a checkpoint's file dtype
differs from the graph dtype or the quantized storage layout is unexpected. B
measures the real model graph and is robust to that. So **prefer B whenever its
estimate is fast** (measured ~1ms on a real graph; graph construction is
O(layer count), not O(hidden size)); fall back to A when the model classes
(dflash draft classes) are unavailable. If A and B ever disagree, **B — the
real model graph — is the authoritative result**. (See "VRAM estimation
validation".)

### VRAM estimation validation

Implemented in `providers/dflash_vram.py` + `tests/test_dflash_vram.py`
(pure-Python synthetic test vectors, plus a real-MLX test that constructs an
actual Llama on disk and compares A vs B on the same model):

- **A == B** for unquantized (bf16/f32) and quantized (4-bit/8-bit) models
  *when B measures the stored bytes* (exact integer sum of array nbytes).
  Empirical: unquant A=361728 == B, quant A=68864 == B; quant < unquant.
- **`count × fixed_bytes` diverges** for quantized models (`count` is the
  unquantized element count, scales/bias omitted) and for non-bf16 dtypes
  (A=361728 vs naive B=180864; quant A=68864 vs naive B=180864). Never use it.
- `compute_bits_per_weight` is float and can round a byte (observed 62720 vs
  62719) — use the exact integer nbytes sum.
- `mlx_lm.load_model(lazy=True)` reads full weights into host RAM (I/O-bound);
  a genuinely fast B builds the graph without reading weight data. Measured:
  `load_model(lazy)` 0.80ms + `weight_bytes` 0.03ms ≈ 0.83ms on a small model.

Test vectors (synthetic specs in `tests/test_dflash_vram.py`) pin the invariant
that A and B both count the *stored* tensors — including quantized U32-packed
weight + F32 scales + bias — so regressions in the dtype mapping or quantized
layout are caught. The real-MLX test is the non-tautological proof (files vs
lazy graph on the same model).

### Caveat

A post-load measurement (`get_peak_memory` / `observability.memory`) is only
useful for *validation*, never for the scheduler's eviction/OOM decision. It
also requires the measurement to run in the same process that loads the model —
and since the dflash server is spawned as a subprocess (item 2), the estimate
must come from the pre-load projection above, not from the spawned process.

## 2. Server lifecycle (startup, clean exit)

dflash-mlx is a **CLI/HTTP server**, not an embeddable in-process API.
`mlx_lm.server._run_http_server` (called by `dflash_mlx.server.serve_forever`)
is private and blocks in the main thread; there is no public handle to stop it
from another thread. So — consistent with how YAALLB already handles
`llama_cpp` and `dwarfstar` — **manage it as a spawned subprocess using
`subprocess` methods (pure Python, no shell)**.

### Startup

```python
self._process = subprocess.Popen(["dflash", "serve", "--model", ref,
                                  "--host", host, "--port", str(port),
                                  # optional: "--draft-model", ...
                                  ], cwd=dflash_dir)
# dflash waits for the model to load BEFORE it starts listening
# (serve_forever calls wait_until_ready() then _run_http_server), so a 200
# from GET {endpoint}/models means the model is resident AND ready.
wait_server_ready(self.endpoint_uri, self._process, "dflash", DFLASH_READY_TIMEOUT)
```

This exactly matches YAALLB's existing `wait_server_ready()` helper
(`abstractions/ready.py`), already used by `llama_cpp.py`/`dwarfstar.py`.

The `dflash serve` CLI (`dflash_mlx/server/config.py build_parser`) exposes
`--model`, `--host`, `--port`, `--draft-model`/`--draft`, metal limits
(`--wired-limit`, `--cache-limit`), quant, diagnostics. Default host/port are
`127.0.0.1:8000` — same as ds4's defaults, so use distinct ports per provider
instance.

### Clean exit — the important nuance

dflash's own teardown lives in `ServerRuntime.serve_forever`'s `finally`
(stops the response generator via `stop_and_join()` and calls
`shutdown_runtime_cache_manager()`, which flushes the SSD-spill L2 prefix
cache). That path only runs on **SIGINT** — `mlx_lm.server._run_http_server`
catches `KeyboardInterrupt` and does `httpd.shutdown()` +
`response_generator.stop_and_join()`. Python does **not** run `finally` on
SIGTERM by default.

So for a genuinely *clean* exit, the YAALLB `unloadModel` should prefer
**SIGINT first** (triggers the graceful HTTP shutdown + L2 cache flush), then
escalate to SIGTERM/SIGKILL on timeout:

```python
def unloadModel(self, model):
    p = self._process
    if p is not None:
        p.send_signal(signal.SIGINT)          # clean: KeyboardInterrupt → httpd.shutdown() + cache flush
        try:
            p.wait(timeout=UNLOAD_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            p.terminate()                      # SIGTERM
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()                       # SIGKILL escalation (stuck in Metal kernel / disk I/O)
                p.wait()
        self._process = None
    self.resident_model = None
    model._loaded = False
```

This mirrors the `process.terminate() → wait(timeout) → kill()` escalation
already in `llama_cpp.py`/`dwarfstar.py`, but starting from SIGINT for the
clean flush.

## Provider integration shape

Matches `DwarfStarProvider`/`LlamaCppProvider`:

- `_type_id = "dflash"`, `single_resident = True` (dflash serves one
  target+draft per process).
- `Model.memory()` → the item-1 **pre-load** projected MiB (Option A:
  safetensors-metadata weight sum + analytical KV/overhead). Never load to
  measure — the scheduler needs this before the model is resident.
- `getOAIModels()` → dflash answers `/v1/models` natively once resident, so the
  default `Provider.getOAIModels()` works (unlike ds4).
- Config keys: `dflash_dir`, `model_ref`, optional `draft_ref`, `host`, `port`,
  `options` (metal limits, quant, diagnostics).

### Caveat for the readiness gate

dflash `serve` refuses to start if no draft model resolves
(`print_startup_banner` raises), and the registry auto-resolves drafts for
supported targets — so `--draft-model` is optional but `--model` is effectively
required.
