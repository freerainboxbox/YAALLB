"""Shared spawned-server readiness gate used by llama_cpp and dwarfstar.

Both providers spawn a single-model server that loads the model at launch, so
a 200 from GET {endpoint}/models means the model is resident and ready. This
helper polls that probe until it answers, the process exits, or a timeout
elapses — so `model.loaded` reflects the downstream actually accepting
requests, not just the process having been spawned.
"""
import time

import httpx

# Poll interval while waiting for the spawned server to accept requests.
READY_POLL_INTERVAL = 0.5


def wait_server_ready(endpoint_uri: str, process, label: str, timeout: float) -> None:
    """Block until a spawned server starts accepting requests on its endpoint.

    Blocks (off the event loop; loadModel runs in a worker thread) until GET
    {endpoint}/models returns 200, the process exits, or timeout elapses. A
    readiness timeout (or server exit) must not orphan the spawned child, so
    callers terminate it on the raised error.
    """
    deadline = time.monotonic() + timeout
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} exited before becoming ready")
        try:
            resp = httpx.get(endpoint_uri + "/models")
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{label} not ready within {timeout}s")
        time.sleep(READY_POLL_INTERVAL)
