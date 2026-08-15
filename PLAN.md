# Implementation Plan: non-streaming override, load-status tracking, LM Studio 404 detection, on_start preload

## Context (what I read)

- `main.py` hosts the OpenAI-compatible surface. `/v1/chat/completions` is **streaming-only**: it 400s any `stream=false` request, then always returns a `StreamingResponse` that (a) sends a prelim SSE event, (b) schedules/loads the model via `SCHEDULER.submit`, (c) forwards to the provider with an internal startup-failure retry loop (`STARTUP_ATTEMPTS=10`, 2s sleeps) that **guarantees** a 200/SSE outcome (converts exhaustion to an SSE error event), (d) relays upstream SSE chunks.
- `scheduling.py` `Scheduler` owns residency/eviction: `select_evictions` picks least-impact residents; `_serve` loads, marks `in_flight`, and appends to `resident`.
- `abstractions/model.py` `Model._loaded` is a plain bool; providers set it inside `loadModel`.
- `providers/llama_cpp.py` & `dwarfstar.py`: `loadModel` does `subprocess.Popen(...)` then immediately sets `model._loaded = True` — **no downstream readiness check**. The spawned server may not accept requests yet, so the first forwarded request fails and the startup loop masks it as a "provider start" issue.
- `providers/lmstudio.py`: `loadModel` posts `/api/v1/models/load`, polls `/api/v1/models` for the model to appear in the loaded list (`_wait_loaded`, up to `LMS_LOAD_MAX_WAIT=600s`), then sets `_loaded=True`. So YAALLB thinks it's ready, but LM Studio can still 404 on `/v1/chat/completions` (its own background load/TTL/eviction race), and the current forward loop treats that 404 as a generic startup failure with a **blind** 10-attempt/20s retry — which can exhaust while LM Studio is still loading, producing an SSE error while LM Studio eventually completes the request "into the void."
- Model-level overrides live in `provider.model_overrides` (per-model dict in config.json), read by `model_overrides_for` in main.py.

---

## Feature 1 — Model-level non-streaming override

### Config/override keys
Two per-model override keys in `model_overrides`:

- **`"allow_non_streaming": true`** — the **server-side** config that permits this model to be *served* non-streaming. It is explicitly a non-default configuration option, distinct from the client-side `stream` request param. Models without it keep the current streaming-only behavior.
- **`"supports_streaming": false`** — an *edge-case declaration* that this model is **physically non-streaming** (e.g. diffusion LLMs). Default undefined → we assume streaming-capable → keep streaming (the default, per "if it is infeasible to determine, simply keep streaming"). **Confirmed** as the determination mechanism: known non-streaming LLMs are manually flagged this way.

### Routing logic (`main.py` `/v1/chat/completions`)
Compute:
```python
supports_streaming = overrides.get("supports_streaming")          # None | False  (None -> assume True)
allow_non_streaming = overrides.get("allow_non_streaming")        # None | True  (None -> False)
client_stream = bool(body.get("stream", True))                     # default True (current behavior)
```

Branch:
- **Client `stream=true`:**
  - `supports_streaming is False` (model can't stream) → `400` (error out; the model physically cannot stream).
  - otherwise (streaming-capable / undetermined) → **existing streaming path** (unchanged).
- **Client `stream=false`:**
  - `allow_non_streaming is True` → **non-streaming direct-forward path** (below).
  - otherwise (`allow_non_streaming` false or undefined) → `400` (current `stream_required` behavior preserved).

So the edge-case non-streaming model (diffusion) is forced to `stream=false` **and** must have `allow_non_streaming=true`, else it errors on both request shapes.

### Non-streaming direct-forward path
This is a *best-effort* path: **no prelim SSE, no startup-failure retry loop, no success guarantee** — exactly "forwards directly to the downstream API but does not guarantee a success code."

1. `provider = await asyncio.to_thread(lookup_model, PROVIDERS, model_id)` (keep off event loop).
2. Resolve `ctx_length` from override/body (same as streaming).
3. `model = await SCHEDULER.submit(model_id, LoadOptions(...))` — blocks until resident/loaded (no SSE prelim possible, so non-streaming just waits). On `ModelNotFound` → `JSONResponse(404)`; on load error → `JSONResponse(503)`.
4. Build `forward_body`: apply non-ctx overrides via `setdefault`, force `forward_body["stream"] = False`.
5. Single-shot forward (no `while` retry loop):
   ```python
   url = provider.endpoint_uri + "/chat/completions"
   async with httpx.AsyncClient(timeout=None) as client:
       upstream = await client.post(url, json=forward_body, headers=provider._auth_headers())
   SCHEDULER.release(model)
   ```
6. Return the upstream response **as-is**: `Response(upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))`. Non-200 is passed through unchanged (no guarantee).

### Provider-note (clarification 3)
`ds4` and `llama_cpp` implement the OAI server spec correctly, so it is safe to assume `stream=false` is allowed for them — the direct-forward path just forwards it. **LM Studio is the main offender** (can 404 on `stream=false` / unloaded); in the direct-forward path its non-200 is passed through as-is, which is the accepted non-guarantee. Since direct-forward still runs `scheduler.submit` first, LM Studio's `loadModel` readiness polling means the model is loaded by forward time, so the race is mostly avoided.

### Tests (tests/test_routing.py)
- `allow_non_streaming=true` + client `stream=false` → forwarded directly; upstream 404/503 returned as-is (assert status + body, no SSE, no retry loop).
- `allow_non_streaming=true` + client `stream=true` → streaming path unchanged.
- `supports_streaming=false` + client `stream=true` → 400.
- `supports_streaming=false` + client `stream=false` + no `allow_non_streaming` → 400.
- `supports_streaming=false` + `allow_non_streaming=true` + client `stream=false` → direct forward.
- No overrides + client `stream=false` → still 400 `stream_required` (existing test passes).

---

## Feature 2 — Track load status for llama_cpp and dwarfstar

### Problem
`loadModel` sets `_loaded=True` immediately after `Popen`, before the spawned server accepts requests. The forward loop then masks "server not ready" as generic startup failure.

### Change
Give each spawned provider a real readiness gate inside `loadModel`, so `model.loaded` reflects the *downstream* accepting requests, not just process spawn:

- `LlamaCppProvider.loadModel`: after `Popen`, poll `GET {endpoint_uri}/models` (llama-server serves it) until 200 or until `process.poll()` is not None; then set `_loaded=True`. Bound by a readiness timeout (`LLAMA_CPP_READY_TIMEOUT`).
- `DwarfStarProvider.loadModel`: after `Popen`, poll `process.poll()` (alive) and probe its endpoint until it answers. **Verified**: running ds4 at `127.0.0.1:8000` answers `GET /v1/models` → 200 with the OAI model list and `stream` in `supported_parameters`, so `GET {endpoint_uri}/models` is a valid readiness probe. Then `_loaded=True`.
- On readiness timeout → raise (so `_serve`/scheduler fails loudly and the model is not marked resident-ready).

### Also track the failing-forward case
Add a per-model load-state so a forward failure attributable to "model not loaded yet" is recorded distinctly from a genuine provider fault:
- Extend `abstractions/model.py` with a `load_state` field (`"loading" | "ready" | "failed"`) alongside `_loaded`, updated by `loadModel`/forward paths.
- In the forward loop (streaming path), when a non-200 is attributable to the spawned server not being ready (llama_cpp/dwarfstar), mark `load_state="loading"` instead of blindly treating it as `startup_failures` — so the retry continues until the readiness gate says `ready`, then forward.

### Tests
- llama_cpp/dwarfstar: `loadModel` blocks until the endpoint responds / process is alive before `loaded` is True; readiness timeout raises.
- Forward-failure sets `load_state` accordingly.

---

## Feature 3 — LM Studio 404 bug: better detection

### Problem restated
The current forward loop treats *any* non-200 upstream as a startup failure and blindly retries (max 10×/20s). LM Studio can 404 `/v1/chat/completions` because its own background load/TTL/eviction isn't reflected in YAALLB's `_loaded`. When the real load takes longer than YAALLB's retry budget, YAALLB SSE-errors while LM Studio eventually completes the request "into the void."

### Fix (deterministic readiness-based detection, no SDK required)
In the streaming forward loop, branch on **provider type + status**:
- LM Studio 404/4XX "still loading / model not loaded": do **not** count it as a startup failure. Instead deterministically wait for readiness: `await asyncio.to_thread(provider.wait_loaded, model_id)` (reuse the existing `/api/v1/models` polling logic from `_wait_loaded`, bounded by `LMS_LOAD_MAX_WAIT`), then retry the forward. This matches the *actual* model load time instead of a fixed 20s budget.
- Genuine connection errors / 5xx: keep the current `startup_failures` retry path (these are true "provider not ready" conditions for spawned servers).
- Expose a small provider helper, e.g. `LMStudioProvider.wait_loaded(model_id)` (the body of the existing `_wait_loaded`, made public and reusable at forward time).

Net effect: a client never sees an SSE error from a 404 that is really "LM Studio still loading"; YAALLB blocks on the real readiness signal and forwards once ready.

### Tests
- Fake upstream returns 404 → forward loop polls readiness (fake `/api/v1/models` shows the model) → retries → succeeds; `startup_failures` is **not** bumped to exhaustion on the 404 path.
- 404 that never becomes ready → bounded SSE error (but distinct code/path, not `provider_start_failed`).

---

## Feature 4 — `on_start` model-level override (`"always"` / `"once"`)

### Config
Per-model override `"on_start": "always" | "once"` (undefined = no preload). `always` = load at startup, **never evict**. `once` = preload immediately, evictable like normal.

### Scheduler changes (`scheduling.py`)
- Add `Scheduler.protected: set[str]` of model ids that must never be evicted (the `"always"` set).
- In `_serve`'s eviction path, exclude protected residents from `select_evictions`:
  ```python
  evictable = [m for m in self.resident if m.descriptor.modelId not in self.protected]
  to_evict = select_evictions(evictable, shortfall)
  ```
  If not enough evictable memory, `select_evictions` raises `RuntimeError("cannot free enough VRAM...")`.

### Startup preload (`main.py` lifespan, async — after `SCHEDULER.start()`)
- Collect preload targets deterministically: iterate `PROVIDERS` in config order, then each provider's `model_overrides` in insertion order; gather model ids with `on_start`, their `on_start` value + `ctx_length` (override `ctx_length` or `DEFAULT_CTX_LENGTH`).
- For each in order:
  - `model = await SCHEDULER.submit(model_id, LoadOptions(ctx_length=...))`
  - `SCHEDULER.release(model)` (preloaded → resident, not in-flight)
  - If `on_start == "always"`: add `model_id` to `SCHEDULER.protected` *before* loading so it can't be evicted.
- **Startup OOM policy (clarification 4, confirmed):** for **both** `"always"` and `"once"`, if a *singular* model's memory exceeds the VRAM budget (cannot fit even after evicting everything), **fail on startup**: emit `log.error(...)` (red) with an error message and **exit with a non-zero code** (abort startup, not warn-and-continue). (A model that fits alone but collides with earlier preloads is handled by normal eviction; only "cannot fit alone" hard-fails startup.)

### Runtime refusal (clarification 4)
If a runtime-requested model is **impossible to load** (its single-model footprint exceeds the budget, so `select_evictions` raises `RuntimeError`), refuse the load and **return an error** — never kill the YAALLB process:
- Streaming path already does this: `_serve` raises → `_run` sets the future's exception → the route catches `Exception` → SSE error event (`model_load_failed`).
- Non-streaming path must mirror it: `scheduler.submit` raising → `JSONResponse(503)` (not an uncaught exception).
- Add a regression test asserting the process/loop survives and an error is returned.

### Tests (tests/test_scheduling.py + test_routing.py)
- `select_evictions`/`_serve` excludes protected models (protected model never evicted even when over budget).
- `once` models are preloaded but still evictable.
- Preload routine loads `always` + `once` in deterministic order and marks `always` protected.
- Startup hard-fails if a singular on_start model exceeds the budget.
- Runtime impossible-load returns an error without killing the process (both streaming SSE + non-streaming JSON).

---

## Optional — LM Studio Python SDK ("convenience API") refactor: pros/cons

I researched `lmstudio-python` (PyPI `lmstudio`) directly. Key facts:
- It is a **client** library, not a server/proxy: it drives LM Studio over a **websocket JSON-RPC** channel (`Client`, `SyncLMStudioWebsocket`, `remote_call("getOrLoad"/"loadModel"/"unloadModel"/"respond"...)`), completely separate from the HTTP OpenAI `/v1` surface.
- It exposes a top-level convenience API: `lms.llm(model_key)`, `lms.Chat(...)`, `model.respond(...)`, `model.respond_stream(...)`, `model.complete(...)`. Loading is `model(model_key, ttl=..., config=..., on_load_progress=...)` which does `getOrLoad` (load-if-not-loaded), plus `load_new_instance` and `unload`. There is a `DEFAULT_TTL` auto-unload and typed exceptions (`LMStudioClientError`, etc.).

### Pros (if refactored)
1. **Directly fixes the 404 race**: `getOrLoad` + TTL + load-progress callbacks give deterministic readiness semantics, replacing the fragile HTTP status-sniffing.
2. **Maintained, typed API**: less custom management-API HTTP/polling code to keep in sync with LM Studio releases; typed exceptions instead of parsing 404 bodies.
3. **Cleaner load/unload control** (progress callbacks, explicit unload), which would help both load-status tracking and `on_start`.

### Cons (why I lean against it)
1. **Architecture mismatch**: YAALLB is an OpenAI-compatible HTTP *reverse proxy*. The SDK is a *client* that talks over websockets. Adopting it for LM Studio means YAALLB must consume the SDK and **re-serialize** chat completions back into OpenAI HTTP JSON/SSE — losing the clean pass-through and adding a translation layer that must stay bug-for-bug compatible with OpenAI wire format.
2. **The convenience API's global state is wrong for a server**: `lms.llm()` implicitly creates a default `Client` that lives until interpreter exit — unsuitable for a long-lived FastAPI process; you'd be forced into the scoped `Client.llm` namespace anyway, which is basically the raw websocket API.
3. **Two transports complicate the shared `Provider` abstraction**: LM Studio (websocket client) vs. llama_cpp/ds4 (HTTP reverse-proxy + spawned processes) would no longer share the same forward path, and the new non-streaming direct-forward feature is HTTP-native.
4. **Heavier dependency** (websocket transport, new Python-version/compat surface) for one provider's bug.
5. **Non-streaming direct-forward** (Feature 1) is explicitly about *forwarding the request as-is* to the downstream API — the SDK path can't give you HTTP pass-through.

### Recommendation
**Do not refactor into the SDK.** Keep the HTTP architecture and fix the 404 detection deterministically (Feature 3) within the existing management-API polling — it reuses code that already exists (`_wait_loaded`), adds no dependency, and preserves the clean proxy design and the new direct-forward feature.

I'd like your call on this before implementation. If you have a strong reason to use the SDK (e.g. you want load-progress/UI integration or the TTL auto-unload), I can revisit — but my default is to stay on HTTP.

---

## Proposed commit order (AGENTS.md: bugfixes first)

1. **Bugfix — LM Studio 404 detection** (Feature 3) + expose `wait_loaded`.
2. **Bugfix — llama_cpp/dwarfstar load-status tracking** (Feature 2).
3. **Feature — non-streaming model override** (Feature 1).
4. **Feature — on_start preload** (Feature 4).

Red-green: write each test, confirm red, then implement, then green.

## Resolved confirmations
1. `"supports_streaming": false` is the determination mechanism — known non-streaming LLMs are manually flagged. ✓
2. Startup OOM = `log.error()` (red) + non-zero exit code (abort startup). ✓
3. ds4 verified: `GET /v1/models` at `127.0.0.1:8000` returns 200 (OAI list, `stream` supported) → valid readiness probe. ✓
