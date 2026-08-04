"""YAALLB server entrypoint.

Runs the FastAPI application. Routes are added as features land; this file
currently only hosts the app and the CLI launcher.
"""

import argparse

from fastapi import FastAPI
import uvicorn

app = FastAPI()


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
