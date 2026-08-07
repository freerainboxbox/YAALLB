## Yet Another Apple LLM Load Balancer

A VRAM-aware LLM load balancer, quick and dirty, to point to your existing LLM runners.

Currently targets LM Studio and antirez/ds4. More may be supported later.

You can specify a VRAM limit on your Apple Silicon Mac in MB, and YAALLB will set `iogpu.wired_limit_mb` to that limit and respect your memory by evicting the least-impact resident models when a new load would exceed the budget.

## Layout

```
main.py            FastAPI app, OpenAI-compatible routes, CLI launcher
scheduling.py      VRAM-aware model scheduler and eviction
log.py             Colored, ISO-timestamped logging to stderr
abstractions/      Base types: Provider, Model, ModelDescriptor, LoadOptions; routing
providers/         Concrete providers: LMStudioProvider, DwarfStarProvider
config.json        Provider instances per type ("lms", "ds4", ...)
tests/             pytest suite
pyproject.toml     Project metadata and dependencies (uv-managed)
```

## Running

Start the server with:

```sh
uv run python main.py
```

The server binds to `127.0.0.1:4343` by default. Configure the bind address
and port via CLI flags; `--address 0.0.0.0` binds everywhere:

```sh
uv run python main.py --address 0.0.0.0 --port 8000
```

Run `uv run python main.py --help` (or pass an invalid command) for the
full CLI documentation.

## Config

Provider instances are configured in `config.json` at the top of the repo
(override the path with `--config`). The format maps a provider `type` to a
list of instance config objects:

```json
{
  "vram_limit_mb": 24576,
  "yaallb": {
    "address": "127.0.0.1",
    "port": 4343,
    "ctx_length": 4096
  },
  "ds4": [ {config for instance 0}, {config for instance 1}, ... ],
  "lms":  [ ... ],
  ... (other provider types, defined later)
}
```

When relevant, providers can have API keys specified at the provider level as `"api_key"`.

The `yaallb` object holds server-level settings: `address` and `port` are the
bind address/port, and `ctx_length` is the default context length used when a
request specifies no `context_length`. `max_tokens` is a client-side constraint
on the number of tokens emitted and is **not** used for context sizing — only
`context_length` (or the default) drives the model's context window. The CLI
flags `--address` and `--port` override these only when explicitly passed;
otherwise config.json is the source of truth.

The position in each list is that instance's `_instance_id`. Types that are
absent are simply disabled. Each instance object is applied on top of the
provider's built-in defaults, so you only need to write the fields you want
to override.

Each provider instance may carry an optional `model_overrides` map, keyed by
model ID, of per-model parameters. `ctx_length` there overrides the default
context length for that model (when the request doesn't set one), and any
other keys (e.g. `temperature`, `top_p`) are injected into the forwarded
request body as defaults when the client didn't specify them:

```json
{
  "lms": [
    {
      "host": "127.0.0.1",
      "port": 1234,
      "model_overrides": {
        "qwen/qwen3-0.6b-mlx": { "ctx_length": 8192, "temperature": 0.7 }
      }
    }
  ]
}
```

`vram_limit_mb` (top-level, default `24576`) is the global VRAM budget in
MiB. When a new load would exceed it, YAALLB picks the **least-impact
eviction set** from the resident models (those reporting `memory() > 0`):

- Candidate A: the smallest single resident model that alone frees enough.
- Candidate B: greedily accumulate resident models smallest-to-next-smallest
  until the freed total covers the shortfall.

Whichever set over-evicts the least (is closest to the shortfall) is chosen,
ties breaking toward the single model. If neither candidate can free enough,
the request fails rather than over-committing. Models reporting `memory() == 0`
(future cloud providers) are never evicted and load without eviction.

Requests for an eviction target are **line-cut** (served first out of the
queue) and **drained** (in-flight I/O completes) before the model is actually
unloaded, so no request is cut off mid-generation.

On startup YAALLB sets the macOS Metal VRAM cap to match `vram_limit_mb` via
`sudo sysctl iogpu.wired_limit_mb=<mb>`. It first reads the current value (no
privileges needed); if it already matches, no write is attempted, so you only
need `sudo` once after a reboot — unless you change `vram_limit_mb`. The write
runs under sudo: YAALLB logs a warning and continues (the software scheduler
still enforces the budget) when sudo is denied or a password is required.

Each provider's `host`/`port` (or `endpoint_uri`) doubles as the reverse-proxy
target: `/v1/chat/completions` schedules a model, then forwards the request
body to `{endpoint_uri}/chat/completions` and relays the upstream response
back. `stream: true` requests are proxied as an SSE stream. The model stays
in-flight (so it isn't evicted) until the upstream reply completes.

`/v1/chat/completions` is **streaming-only**:

- Requests that don't set `stream: true` are refused with a `400` error
  (`code: stream_required`) rather than silently downgraded.
- Every streaming request gets an immediate **prelim SSE event**
  (`{"status": "processing", "model": ..., "choices": []}`) so the client sees
  a `200` and knows YAALLB is awake before the scheduler finishes a
  potentially long model load. A `200` here is **not** confirmation that a
  full reply will come: if the downstream provider fails, the failure is
  delivered as an **SSE error event** within the same `200` stream — never as
  a `503`/`500`/4XX body, which the client would just reject.

When a provider isn't ready yet (e.g. ds4-server is still starting up), the
forward is **retried internally** up to `STARTUP_ATTEMPTS` (10) times, then an
SSE error event (`code: provider_start_failed`) is emitted. Each failure bumps
a per-provider startup counter, and the counter resets once the provider
serves a request successfully.

### Providers

#### ds4

`ds4` is spawned and terminated by YAALLB (it has no native load/unload), so
each instance needs to know how to launch `ds4-server` from a working
directory.

| key          | default        | required                                                                 |
| ------------ | -------------- | ------------------------------------------------------------------------ |
| `ds4_dir`    | —              | yes — path to your ds4 build; the working directory the server runs from |
| `gguf_path`  | —              | yes — path to the GGUF model loaded by ds4 (relative to `ds4_dir`)       |
| `host`       | `127.0.0.1`    | no — ds4 bind address, also the reverse-proxy target                     |
| `port`       | `8000`         | no — ds4 bind port, also the reverse-proxy target                        |
| `binary`     | `./ds4-server` | no — program to run, relative to `ds4_dir`                               |
| `options`    | `{}`           | no — overrides for ds4-server flags (see below)                          |
| `ctx_length` | —              | no — provider-level context length, overrides the per-model one          |

`ctx_length` is available as a provider-level override,
and is a key in the provider object (see usage below) rather than model-level (so NOT in the "options" key).
ds4 sets `--ctx` once at startup and both of its served models
(`deepseek-v4-flash`, `deepseek-v4-pro`) inherit it, so the provider-level
`ctx_length` (when set) overrides any per-model `ctx_length` and is what
`/v1/models` reports for both models.

`options` keys are the ds4-server flag names with dashes turned into
underscores. A flag is only emitted when its value differs from the default
shown below (booleans only when `true`), so `ds4-server` applies its own
defaults for everything you don't set.

| options key                          | flag                                   | kind  | default |
| ------------------------------------ | -------------------------------------- | ----- | ------- |
| `backend`                            | `--backend`                            | value | —       |
| `metal`                              | `--metal`                              | flag  | false   |
| `cuda`                               | `--cuda`                               | flag  | false   |
| `cpu`                                | `--cpu`                                | flag  | false   |
| `gpu_vram`                           | `--gpu-vram`                           | value | —       |
| `gpu_devices`                        | `--gpu-devices`                        | value | —       |
| `cuda_tensor_parallel`               | `--cuda-tensor-parallel`               | flag  | false   |
| `tokens`                             | `-n`                                   | value | —       |
| `threads`                            | `-t`                                   | value | —       |
| `power`                              | `--power`                              | value | 100     |
| `ssd_streaming`                      | `--ssd-streaming`                      | flag  | false   |
| `ssd_streaming_cold`                 | `--ssd-streaming-cold`                 | flag  | false   |
| `ssd_streaming_cache_experts`        | `--ssd-streaming-cache-experts`        | value | —       |
| `ssd_streaming_full_layers`          | `--ssd-streaming-full-layers`          | value | —       |
| `ssd_streaming_preload_experts`      | `--ssd-streaming-preload-experts`      | value | —       |
| `simulate_used_memory`               | `--simulate-used-memory`               | value | —       |
| `prefill_chunk`                      | `--prefill-chunk`                      | value | —       |
| `cors`                               | `--cors`                               | flag  | false   |
| `trace`                              | `--trace`                              | value | —       |
| `batched_session`                    | `--batched-session`                    | value | —       |
| `kv_disk_dir`                        | `--kv-disk-dir`                        | value | —       |
| `kv_disk_space_mb`                   | `--kv-disk-space-mb`                   | value | 4096    |
| `kv_cache_min_tokens`                | `--kv-cache-min-tokens`                | value | 512     |
| `kv_cache_cold_max_tokens`           | `--kv-cache-cold-max-tokens`           | value | 30000   |
| `kv_cache_continued_interval_tokens` | `--kv-cache-continued-interval-tokens` | value | 10000   |
| `kv_cache_boundary_trim_tokens`      | `--kv-cache-boundary-trim-tokens`      | value | 32      |
| `kv_cache_boundary_align_tokens`     | `--kv-cache-boundary-align-tokens`     | value | 2048    |
| `kv_cache_reject_different_quant`    | `--kv-cache-reject-different-quant`    | flag  | false   |
| `disable_exact_dsml_tool_replay`     | `--disable-exact-dsml-tool-replay`     | flag  | false   |
| `tool_memory_max_ids`                | `--tool-memory-max-ids`                | value | 100000  |

For example, the manual command

```sh
./ds4-server -m ./ds4flash-0731.gguf --kv-disk-dir /tmp/ds4-0731-kv \
  --kv-disk-space-mb 262144 --ctx 1000000
```

is configured as:

```json
{
  "ds4": [
    {
      "ds4_dir": "/path/to/ds4",
      "gguf_path": "./ds4flash-0731.gguf",
      "host": "127.0.0.1",
      "port": 8000,
      "options": {
        "kv_disk_dir": "/tmp/ds4-0731-kv",
        "kv_disk_space_mb": 262144
      },
      "ctx_length": 1000000
    }
  ]
}
```

#### lms

LM Studio serves its own API natively; instances only need `host` and
`port` (defaults `127.0.0.1` and `1234`), plus `api_key` when server
authentication is enabled.

YAALLB drives LM Studio through its management REST API (`/api/v1`):
`getModelsDescriptors` lists LLMs from `GET /api/v1/models`, `loadModel`
posts to `POST /api/v1/models/load` (with `context_length`), and `unloadModel`
posts to `POST /api/v1/models/unload`. VRAM estimates still come from the
`lms` CLI (`--estimate-only`). This is a limitation of the LM Studio API.

LM Studio blocks on `POST /api/v1/models/load` until the model finishes
loading, so YAALLB gives that call a long timeout (`LMS_LOAD_TIMEOUT`, 300s)
and then **polls** `GET /api/v1/models` until the model key appears as a
loaded LLM (`LMS_LOAD_MAX_WAIT`, 600s). A load timeout or LM Studio's
"still loading" 4XX is treated as still-in-progress and never forwarded to
the client as an error; only a true failure (auth `401`, 5XX, or the model
never becoming ready within the deadline) raises.

## Graceful shutdown

On exit (Ctrl-C/SIGTERM), YAALLB flushes queued and in-flight requests, then
unloads every resident model: LM Studio instances get the unload API route
called, and ds4 instances simply terminate their spawned server process.

## Roadmap

llama.cpp, mlx-lm, mlx-vlm, oMLX, and cloud API providers should be supported eventually.
