"""YAALLB server entrypoint.

Hosts the OpenAI-compatible surface: /v1/chat/completions decodes the model
ID, schedules a provider/model under a VRAM budget, and reverse-proxies the
request to the resolved provider; /v1/models aggregates the model lists of
every provider.
"""

import argparse
import asyncio
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from starlette.responses import JSONResponse, StreamingResponse

import log
from abstractions.load_options import LoadOptions
from abstractions.provider import Provider
from abstractions.routing import lookup_model
from providers.dwarfstar import DwarfStarProvider
from providers.lmstudio import LMStudioProvider
from scheduling import ModelNotFound, Scheduler
import uvicorn

DEFAULT_VRAM_LIMIT_MIB = 24576
DEFAULT_CTX_LENGTH = 4096
STARTUP_ATTEMPTS = 10

DEFAULT_YAALLB_CONFIG = {
    "address": "127.0.0.1",
    "port": 4343,
    "ctx_length": DEFAULT_CTX_LENGTH,
}

PROVIDER_TYPES = {
    "lms": LMStudioProvider,
    "ds4": DwarfStarProvider,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SCHEDULER is not None:
        await SCHEDULER.start()
    yield
    if SCHEDULER is not None:
        await SCHEDULER.stop()
        for model in SCHEDULER.resident:
            try:
                model.unloadModel()
                log.info(
                    f"shutdown unloaded model={model.descriptor.modelId} "
                    f"provider={model.descriptor.provider._type_id}"
                    f"#{getattr(model.descriptor.provider, '_instance_id', 0)}"
                )
            except Exception as e:
                log.error(
                    f"shutdown failed to unload "
                    f"model={model.descriptor.modelId}: {e}"
                )


app = FastAPI(lifespan=lifespan)

# Populated at runtime from config.json; tests inject fakes.
PROVIDERS: list[Provider] = []
SCHEDULER: Scheduler | None = None


def load_providers(config_path: str) -> list[Provider]:
    providers = []
    if not Path(config_path).exists():
        return providers
    with open(config_path) as f:
        config = json.load(f)
    for type_id, instances in config.items():
        provider_cls = PROVIDER_TYPES.get(type_id)
        if provider_cls is None:
            continue
        for instance_id, instance_config in enumerate(instances):
            providers.append(provider_cls(instance_id, instance_config))
    return providers


def load_vram_limit(config_path: str) -> int:
    if not Path(config_path).exists():
        return DEFAULT_VRAM_LIMIT_MIB
    with open(config_path) as f:
        config = json.load(f)
    return config.get("vram_limit_mb", DEFAULT_VRAM_LIMIT_MIB)


def load_yaallb_config(config_path: str) -> dict:
    """Read the server-level settings (bind address, port, default ctx_length).

    Falls back to defaults for keys that are absent.
    """
    cfg = dict(DEFAULT_YAALLB_CONFIG)
    if not Path(config_path).exists():
        return cfg
    with open(config_path) as f:
        config = json.load(f)
    cfg.update(config.get("yaallb", {}))
    return cfg


def model_overrides_for(provider: Provider, model_id: str) -> dict:
    """Per-model overrides configured on a provider, or {} if none."""
    overrides = getattr(provider, "model_overrides", None) or {}
    return overrides.get(model_id, {}) or {}


def read_iogpu_wired_limit() -> int | None:
    """Read the current macOS Metal VRAM cap in MiB (no privileges needed).

    Returns None if the sysctl is unavailable or cannot be read.
    """
    sysctl = shutil.which("sysctl")
    if sysctl is None:
        return None
    try:
        proc = subprocess.run(
            [sysctl, "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def set_iogpu_wired_limit(vram_limit_mb: int) -> bool:
    """Set the macOS Metal VRAM cap to match YAALLB's budget.

    The kernel sysctl is privileged, so the write runs under sudo. Returns
    True on success. On failure logs a warning; YAALLB's own scheduler still
    enforces the budget in software regardless.
    """
    current = read_iogpu_wired_limit()
    if current == vram_limit_mb:
        log.info(
            f"iogpu.wired_limit_mb already {vram_limit_mb}; no sudo needed"
        )
        return True
    log.info(
        f"iogpu.wired_limit_mb={current if current is not None else '?'}, "
        f"target={vram_limit_mb} -> requesting sudo write"
    )
    sysctl = shutil.which("sysctl")
    sudo = shutil.which("sudo")
    if sysctl is None:
        log.warning("sysctl unavailable; cannot set iogpu.wired_limit_mb")
        return False
    if sudo is None:
        log.warning("sudo unavailable; cannot set iogpu.wired_limit_mb")
        return False
    try:
        proc = subprocess.run(
            [sudo, sysctl, "-w", f"iogpu.wired_limit_mb={vram_limit_mb}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        log.warning(f"failed to set iogpu.wired_limit_mb: {e}")
        return False
    if proc.returncode != 0:
        log.warning(
            f"could not set iogpu.wired_limit_mb={vram_limit_mb} "
            f"(sudo denied or requires a password): {proc.stderr.strip()}"
        )
        return False
    log.info(f"set iogpu.wired_limit_mb={vram_limit_mb}")
    return True


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def _sse_error(code: str, message: str) -> bytes:
    return _sse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        }
    )


def _bump_startup_failures(provider: Provider, detail: str) -> int:
    failures = getattr(provider, "startup_failures", 0) + 1
    provider.startup_failures = failures
    log.warning(
        f"provider {provider.endpoint_uri} not ready "
        f"({failures}/{STARTUP_ATTEMPTS}): {detail}; retrying"
    )
    return failures


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    model_id = body.get("model")

    # The endpoint is streaming-only: a non-200 body would be rejected by the
    # client, so refuse non-streaming requests up front instead of silently
    # downgrading them.
    if not body.get("stream", False):
        log.warning(f"chat request model={model_id} rejected: stream required")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        "streaming must be enabled (stream=true) for this "
                        "endpoint to work"
                    ),
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "stream_required",
                }
            },
        )

    provider = lookup_model(PROVIDERS, model_id)
    overrides = model_overrides_for(provider, model_id) if provider else {}
    default_ctx = overrides.get("ctx_length") or DEFAULT_CTX_LENGTH
    ctx_length = body.get("context_length") or default_ctx

    async def event_stream():
        # Prelim event so the client sees a 200 and knows YAALLB is awake
        # before the scheduler finishes a potentially long model load.
        yield _sse({"status": "processing", "model": model_id, "choices": []})

        try:
            model = await SCHEDULER.submit(
                model_id, LoadOptions(ctx_length=ctx_length)
            )
        except ModelNotFound:
            log.error(f"model not found: {model_id}")
            yield _sse_error(
                "model_not_found",
                f"The model `{model_id}` does not exist or you do not have "
                "access to it.",
            )
            return
        except Exception as e:
            # Any other scheduler-side failure (VRAM exhaustion, memory
            # estimate, load error) must become an SSE error event too — the
            # prelim 200 is already committed, so an uncaught exception would
            # abort the stream with no error event.
            log.error(f"model load failed: {e}")
            yield _sse_error(
                "model_load_failed",
                f"failed to load model `{model_id}`: {e}",
            )
            return

        # Apply non-ctx model overrides (temperature, top_p, ...) as defaults
        # when the client didn't specify them; they ride along in the body.
        forward_body = dict(body)
        for key, value in overrides.items():
            if key == "ctx_length":
                continue
            forward_body.setdefault(key, value)
        forward_body["stream"] = True

        provider = model.descriptor.provider
        log.info(
            f"chat request model={model_id} "
            f"provider={provider._type_id}#{getattr(provider, '_instance_id', 0)} "
            f"stream=true ctx={ctx_length}"
        )

        url = provider.endpoint_uri + "/chat/completions"
        headers = provider._auth_headers()

        # Forward as an SSE stream, retrying transient failures internally
        # (bounded by STARTUP_ATTEMPTS) instead of surfacing a 503/500 or an
        # LM Studio "still loading" 4XX. Exhaustion becomes an SSE error event.
        while True:
            client = httpx.AsyncClient(timeout=None)
            try:
                upstream = await client.send(
                    client.build_request(
                        "POST", url, json=forward_body, headers=headers
                    ),
                    stream=True,
                )
            except httpx.HTTPError as e:
                await client.aclose()
                failures = _bump_startup_failures(provider, f"connection error: {e}")
            else:
                if upstream.status_code != 200:
                    await client.aclose()
                    failures = _bump_startup_failures(
                        provider, f"upstream status {upstream.status_code}"
                    )
                else:
                    provider.startup_failures = 0
                    break

            if failures < STARTUP_ATTEMPTS:
                await asyncio.sleep(2)
                continue
            log.error(
                f"provider {provider.endpoint_uri} failed to start "
                f"after {STARTUP_ATTEMPTS} attempts"
            )
            SCHEDULER.release(model)
            yield _sse_error(
                "provider_start_failed",
                f"provider {provider.endpoint_uri} failed to start",
            )
            return

        # Relay the upstream SSE stream chunk-for-chunk.
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await client.aclose()
            SCHEDULER.release(model)

    return StreamingResponse(
        event_stream(),
        status_code=200,
        media_type="text/event-stream",
    )


@app.get("/v1/models")
def list_models() -> dict:
    data = []
    for provider in PROVIDERS:
        data.extend(provider.getOAIModels())
    return {"object": "list", "data": data}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yaallb",
        description="Yet Another Apple LLM Load Balancer",
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="Address to bind. Use 0.0.0.0 to bind everywhere. (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4343,
        help="Port to listen on. (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.json"),
        help="Path to config.json. (default: %(default)s)",
    )
    args = parser.parse_args()

    yaallb = load_yaallb_config(args.config)

    global SCHEDULER, PROVIDERS
    providers = load_providers(args.config)
    PROVIDERS.extend(providers)
    vram_limit_mb = load_vram_limit(args.config)
    SCHEDULER = Scheduler(PROVIDERS, vram_limit_mb)

    # CLI flags override config only when explicitly passed (differs from the
    # CLI defaults); otherwise config.json is the source of truth.
    address = yaallb["address"] if args.address == "127.0.0.1" else args.address
    port = yaallb["port"] if args.port == 4343 else args.port
    global DEFAULT_CTX_LENGTH
    DEFAULT_CTX_LENGTH = yaallb["ctx_length"]

    set_iogpu_wired_limit(vram_limit_mb)

    log.info(
        f"starting yaallb on {address}:{port} "
        f"providers={[p._type_id for p in providers]} "
        f"vram_limit={vram_limit_mb} MiB default_ctx={yaallb['ctx_length']}"
    )

    uvicorn.run(app, host=address, port=port)


if __name__ == "__main__":
    main()
