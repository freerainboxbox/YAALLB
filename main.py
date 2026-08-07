"""YAALLB server entrypoint.

Hosts the OpenAI-compatible surface: /v1/chat/completions decodes the model
ID, schedules a provider/model under a VRAM budget, and reverse-proxies the
request to the resolved provider; /v1/models aggregates the model lists of
every provider.
"""

import argparse
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Response
from starlette.responses import StreamingResponse

import log
from abstractions.load_options import LoadOptions
from abstractions.provider import Provider
from abstractions.routing import lookup_model
from providers.dwarfstar import DwarfStarProvider
from providers.lmstudio import LMStudioProvider
from scheduling import ModelNotFound, Scheduler
import uvicorn

DEFAULT_VRAM_LIMIT_MIB = 24576

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


def model_not_found_error(model_id: str) -> dict:
    return {
        "error": {
            "message": (
                f"The model `{model_id}` does not exist or you do not have "
                "access to it."
            ),
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


async def _forward_chat(provider: Provider, body: dict, stream: bool):
    """Proxy /v1/chat/completions to the provider.

    Returns (response_or_generator, is_stream). On transport failure raises
    httpx.HTTPError so the caller can 502.
    """
    url = provider.endpoint_uri + "/chat/completions"
    headers = provider._auth_headers()
    client = httpx.AsyncClient(timeout=None)
    if stream:
        req = client.build_request("POST", url, json=body, headers=headers)
        upstream = await client.send(req, stream=True)
        return client, upstream
    try:
        return await client.post(url, json=body, headers=headers)
    finally:
        await client.aclose()


@app.post("/v1/chat/completions")
async def chat_completions(body: dict, response: Response):
    model_id = body.get("model")
    ctx_length = body.get("context_length") or body.get("max_tokens") or 4096
    try:
        model = await SCHEDULER.submit(model_id, LoadOptions(ctx_length=ctx_length))
    except ModelNotFound:
        log.error(f"model not found: {model_id}")
        response.status_code = 404
        return model_not_found_error(model_id)

    provider = model.descriptor.provider
    stream = body.get("stream", False)
    log.info(
        f"chat request model={model_id} "
        f"provider={provider._type_id}#{getattr(provider, '_instance_id', 0)} "
        f"stream={stream} ctx={ctx_length}"
    )
    try:
        result = await _forward_chat(provider, body, stream)
    except httpx.HTTPError:
        log.error(f"upstream {provider.endpoint_uri} unreachable")
        SCHEDULER.release(model)
        response.status_code = 502
        return {
            "error": {
                "message": f"upstream {provider.endpoint_uri} unreachable",
                "type": "api_error",
                "param": None,
                "code": "upstream_unreachable",
            }
        }

    if stream:
        client, upstream = result
        if upstream.status_code != 200:
            log.error(
                f"upstream {provider.endpoint_uri} "
                f"error status={upstream.status_code}"
            )
            SCHEDULER.release(model)
            await client.aclose()
            response.status_code = upstream.status_code
            return await upstream.aread()

        async def event_stream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await client.aclose()
                SCHEDULER.release(model)

        return StreamingResponse(
            event_stream(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    upstream = result
    if upstream.status_code != 200:
        log.error(
            f"upstream {provider.endpoint_uri} error status={upstream.status_code}"
        )
    SCHEDULER.release(model)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
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

    global SCHEDULER, PROVIDERS
    providers = load_providers(args.config)
    PROVIDERS.extend(providers)
    vram_limit_mb = load_vram_limit(args.config)
    SCHEDULER = Scheduler(PROVIDERS, vram_limit_mb)

    set_iogpu_wired_limit(vram_limit_mb)

    log.info(
        f"starting yaallb on {args.address}:{args.port} "
        f"providers={[p._type_id for p in providers]} "
        f"vram_limit={vram_limit_mb} MiB"
    )

    uvicorn.run(app, host=args.address, port=args.port)


if __name__ == "__main__":
    main()
