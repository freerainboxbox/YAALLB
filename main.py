"""YAALLB server entrypoint.

Hosts the OpenAI-compatible surface: /v1/chat/completions decodes the model
ID and routes it to the serving provider; /v1/models aggregates the model
lists of every provider. Reverse proxy forwarding is not wired yet.
"""

import argparse

import httpx
from fastapi import FastAPI, Response

from abstractions.provider import Provider
from abstractions.routing import lookup_model
import uvicorn

app = FastAPI()

# Populated at runtime once providers are configured; tests inject fakes.
PROVIDERS: list[Provider] = []


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


@app.post("/v1/chat/completions")
def chat_completions(body: dict, response: Response) -> dict:
    model_id = body.get("model")
    provider = lookup_model(PROVIDERS, model_id)
    if provider is None:
        response.status_code = 404
        return model_not_found_error(model_id)
    # Reverse proxy not wired yet; report the routing decision.
    return {"model": model_id, "provider": provider.endpoint_uri, "status": "routed"}


@app.get("/v1/models")
def list_models() -> dict:
    data = []
    for provider in PROVIDERS:
        resp = httpx.get(provider.endpoint_uri + "/models")
        resp.raise_for_status()
        data.extend(resp.json().get("data", []))
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
    args = parser.parse_args()

    uvicorn.run(app, host=args.address, port=args.port)


if __name__ == "__main__":
    main()
