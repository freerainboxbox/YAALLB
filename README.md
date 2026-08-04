## Yet Another Apple LLM Load Balancer

A VRAM-aware LLM load balancer, quick and dirty, to point to your existing LLM runners.

Currently targets LM Studio and antirez/ds4. More may be supported later.

You can specify a VRAM limit on your Apple Silicon Mac in MB, and YAALLB will set `iogpu.wired_limit_mb` to that limit and respect your memory by unloading the least recently used model when asked.

## Layout

```
main.py            FastAPI app, OpenAI-compatible routes, CLI launcher
abstractions/      Base types: Provider, Model, ModelDescriptor, LoadOptions; routing
providers/         Concrete providers: LMStudioProvider, DwarfStarProvider
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