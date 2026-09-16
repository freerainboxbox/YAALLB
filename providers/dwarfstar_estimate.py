"""Projected memory footprint of a spawned ds4-server, computed by ds4 itself.

Why not a formula: ds4's context/KV/scratch footprint depends on the model
shape recorded in the GGUF (DeepSeek V4 Flash vs Pro vs GLM all differ in layer
count, head dim, and per-layer compression ratios), on the backend, on the
effective prefill chunk, and on SSD streaming. It is ds4's own arithmetic, used
by ds4 for its own startup logs ("context buffers ... MiB"), so YAALLB asks for
it instead of re-deriving it. See the "VRAM footprint" part of the ds4 provider
section in README.md, plus tools/ds4_estimate.c (the estimator program, built
inside the ds4 tree by tools/ds4-estimate.mk).

Unlike the dflash-mlx provider, ds4 is *not* a cached-VRAM-estimate provider
(see README "Cached VRAM estimates"): one estimator run costs well under a
second, its result cannot be split into ctx-independent components (the whole
context term is shape-dependent), and `Model.memory()` is called with a
ctx_length that a request may change. So results are memoized per-process and
warmed at startup (`warm_dwarfstar_estimates`) instead of being persisted under
the config-hash cache file.
"""

import itertools
import json
import os
import subprocess
from pathlib import Path

import log

# Schema version this module can read (tools/ds4_estimate.c prints it).
# Schema 3 added ds4's own model identity (family/id/aliases) plus
# spec_graph_supported; an older helper cannot say which model a GGUF is, so it
# is refused with a rebuild hint rather than scheduled under DeepSeek's shape.
DS4_ESTIMATOR_SCHEMA_VERSION = 3

# Per-run ds4 instance lock, so estimating never collides with a running
# ds4-server (which holds /tmp/ds4.lock by default) and never blocks one.
DS4_LOCK_DIR = Path("/tmp/yaallb")

# Engine open maps the GGUF and parses its metadata; allow slow disks.
DS4_ESTIMATOR_TIMEOUT = 60

# Fixed process overhead on top of the GGUF bytes plus ds4's own context
# estimate: vocab tables, server/HTTP scaffolding, KV/tool bookkeeping, and the
# graph allocations ds4 makes outside the context estimate. Its value is the
# residual of the flat formula this replaced (83065.32 MiB projected for
# ds4flash-0731.gguf, whose GGUF is 82702.74 MiB), i.e. the overhead that
# YAALLB has been budgeting beyond GGUF + context bytes all along. Nobody has
# measured it per-build since; if your ds4 build differs materially, add the
# difference with `safety_buffer_mib` instead of editing this constant.
DS4_PROCESS_OVERHEAD_MIB = 362.58

# Fallback-only term, used when the estimator binary cannot run. Large-context
# DeepSeek V4 Flash slope on the Metal backend (context bytes per context
# token); small contexts and other backends/shapes deviate from it, which is
# exactly why the estimator exists.
DS4_CTX_BYTES_PER_TOKEN = 16416

# Fallback-only per-session term for a configured drafter: DSpark target-hidden
# capture buffers, verifier snapshots, draft logits and draft-side host buffers,
# none of which follow from file sizes. Measured with the current ds4 tree on
# DeepSeek V4 Flash + its 0731 DSpark support GGUF (3 stages, 3 target layers,
# block 5) at ctx 1000000 on Metal: 219 MiB capture + 86.6 MiB verifier graph +
# 1 MiB host. Context-independent except for the per-stage draft raw cache, so
# it is a much flatter term than the context slope. The estimator reports it
# exactly (`spec_graph_bytes`), and this number is only a stand-in for when it
# cannot run; prefer rebuilding the estimator over trusting it.
DS4_DRAFTER_SCRATCH_FALLBACK_MIB = 306.65

_ESTIMATE_CACHE: dict[tuple, dict] = {}
_WARNED: set[tuple] = set()
_LOCK_SEQ = itertools.count()


class Ds4EstimatorError(RuntimeError):
    """The ds4 estimator could not produce an estimate."""


def build_hint(ds4_dir: str) -> str:
    """The command that produces the estimator binary for a ds4 tree."""
    here = Path(__file__).resolve().parent.parent / "tools" / "ds4-estimate.mk"
    return f"make -C {ds4_dir} -f {here}"


def run_estimator(
    *,
    ds4_dir: str,
    gguf_path: str,
    ctx: int,
    binary: str = "./ds4-estimate",
    backend: str | None = None,
    prefill_chunk: int | None = None,
    ssd_streaming: bool = False,
    mtp_model: str | None = None,
    vision: str | None = None,
    dspark: bool = False,
    mtp: bool = False,
) -> dict:
    """Ask the ds4 tree for its footprint components (bytes).

    Paths are passed through verbatim and resolved against `ds4_dir` (the
    estimator is spawned with that as its cwd, exactly like ds4-server), so a
    relative `gguf_path` means the same thing it means in config.json.

    Raises Ds4EstimatorError when the estimator is missing, fails, or prints
    something this module cannot read. A missing/stale estimator says how to
    build it; a failing one reports ds4's own complaint (bad model path, and so
    on) so the two are distinguishable in the log.
    """
    if not gguf_path:
        # Nothing to measure, so nothing to run: ds4 has no shape to read and
        # the caller should get the (zero-GGUF) fallback rather than a
        # subprocess error about an empty --model.
        raise Ds4EstimatorError("ds4 provider has no gguf_path configured")

    argv = [binary, "--model", gguf_path, "--ctx", str(int(ctx))]
    if backend:
        argv += ["--backend", backend]
    if prefill_chunk:
        argv += ["--prefill-chunk", str(int(prefill_chunk))]
    if ssd_streaming:
        argv.append("--ssd-streaming")
    if mtp_model:
        argv += ["--mtp-model", mtp_model]
    if dspark:
        argv.append("--dspark")
    if mtp:
        # Qwen3.8 Flash Next and GLM 5.3 keep their MTP drafter inside the main
        # GGUF, so no --mtp-model path reveals that a drafter is configured.
        argv.append("--mtp")
    if vision:
        argv += ["--vision", vision]

    DS4_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = DS4_LOCK_DIR / f"ds4-estimate-{os.getpid()}-{next(_LOCK_SEQ)}.lock"
    env = dict(os.environ)
    env["DS4_LOCK_FILE"] = str(lock)
    try:
        proc = subprocess.run(
            argv,
            cwd=ds4_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=DS4_ESTIMATOR_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise Ds4EstimatorError(
            f"ds4 estimator {binary} not found in {ds4_dir}; build it with: "
            f"{build_hint(ds4_dir)}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise Ds4EstimatorError(
            f"ds4 estimator timed out after {DS4_ESTIMATOR_TIMEOUT}s "
            f"(model={gguf_path} ctx={ctx})"
        ) from exc
    finally:
        lock.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise Ds4EstimatorError(
            f"ds4 estimator exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[-500:]}"
        )
    try:
        estimate = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise Ds4EstimatorError(
            f"ds4 estimator printed no JSON; it may be stale — rebuild with: "
            f"{build_hint(ds4_dir)}"
        ) from exc

    if not isinstance(estimate, dict):
        raise Ds4EstimatorError(
            f"ds4 estimator printed {type(estimate).__name__}, not an object; "
            f"rebuild it with: {build_hint(ds4_dir)}"
        )

    version = estimate.get("version")
    if version != DS4_ESTIMATOR_SCHEMA_VERSION:
        raise Ds4EstimatorError(
            f"ds4 estimator schema {version!r} unsupported (want "
            f"{DS4_ESTIMATOR_SCHEMA_VERSION}); rebuild it with: "
            f"{build_hint(ds4_dir)}"
        )
    missing = [
        key
        for key in (
            "model_bytes",
            "support_bytes",
            "vision_bytes",
            "context_bytes",
            # Per-session drafter graph scratch plus its breakdown; absent from
            # schema 1 helpers, whose rebuild hint the error below gives.
            "dspark_capture_bytes",
            "verifier_scratch_bytes",
            "host_scratch_bytes",
            "spec_graph_bytes",
            # The model identity schema 3 added; a helper missing any of them
            # is a schema 2 helper, whose rebuild hint the error below gives.
            "model_id",
        )
        if not isinstance(estimate.get(key), int) or estimate[key] < 0
    ]
    for key in ("model_family", "model_aliases", "spec_graph_supported"):
        value = estimate.get(key)
        if key == "model_family" and not (isinstance(value, str) and value):
            missing.append(key)
        elif key == "model_aliases" and not (
            isinstance(value, list)
            and value
            and all(isinstance(alias, str) and alias for alias in value)
        ):
            missing.append(key)
        elif key == "spec_graph_supported" and not isinstance(value, bool):
            missing.append(key)
    if missing:
        raise Ds4EstimatorError(
            f"ds4 estimator output is missing {missing}; rebuild it with: "
            f"{build_hint(ds4_dir)}"
        )
    return estimate


def gguf_bytes(ds4_dir: str, *paths: str | None) -> int:
    """Bytes of the mapped GGUFs (main model plus support/vision GGUFs).

    Relative paths resolve against `ds4_dir`, the way ds4-server resolves them.
    """
    total = 0
    for path in paths:
        if not path:
            continue
        full = path if os.path.isabs(path) else os.path.join(ds4_dir, path)
        try:
            total += os.path.getsize(full)
        except OSError:
            continue
    return total


def fallback_estimate(
    *,
    ds4_dir: str,
    gguf_path: str,
    ctx: int,
    mtp_model: str | None = None,
    vision: str | None = None,
    dspark: bool = False,
) -> dict:
    """Size-aware estimate for when the ds4 estimator cannot run.

    Uses the real GGUF bytes (the dominant term) plus a flat
    ``DS4_CTX_BYTES_PER_TOKEN`` context term. Only the context term is
    approximate, and only for shapes/backends/contexts that deviate from the
    DeepSeek V4 Flash Metal slope it was taken from. A configured drafter also
    gets ``DS4_DRAFTER_SCRATCH_FALLBACK_MIB`` per session, since none of that
    scratch can be read off file sizes.
    """
    context_bytes = DS4_CTX_BYTES_PER_TOKEN * max(int(ctx), 1)
    model_bytes = gguf_bytes(ds4_dir, gguf_path)
    support_bytes = gguf_bytes(ds4_dir, mtp_model)
    return {
        "source": "fallback",
        "model_name": None,
        "backend": None,
        # No identity without ds4: the provider keeps its configured model list
        # and says so, rather than inventing a shape it did not measure.
        "model_family": None,
        "model_id": None,
        "model_aliases": None,
        "spec_graph_supported": None,
        "ctx": int(ctx),
        "model_bytes": model_bytes,
        "support_bytes": support_bytes,
        "vision_bytes": gguf_bytes(ds4_dir, vision),
        "context_bytes": context_bytes,
        "spec_graph_bytes": int(DS4_DRAFTER_SCRATCH_FALLBACK_MIB * 2**20)
        if (mtp_model or dspark)
        else 0,
    }


def _signature(
    ds4_dir, gguf_path, ctx, binary, backend, prefill_chunk, ssd_streaming,
    mtp_model, vision, dspark, mtp,
) -> tuple:
    return (
        ds4_dir,
        gguf_path,
        int(ctx),
        binary,
        backend,
        prefill_chunk,
        bool(ssd_streaming),
        mtp_model,
        vision,
        bool(dspark),
        bool(mtp),
    )


# Bounded so a router that keeps seeing fresh ctx sizes cannot grow with
# request history; the oldest entries are dropped first.
_CACHE_LIMIT = 256


def _cache_put(key: tuple, result: dict) -> None:
    if len(_ESTIMATE_CACHE) >= _CACHE_LIMIT:
        _ESTIMATE_CACHE.pop(next(iter(_ESTIMATE_CACHE)))
    _ESTIMATE_CACHE[key] = result


def estimate(
    *,
    ds4_dir: str,
    gguf_path: str,
    ctx: int,
    binary: str = "./ds4-estimate",
    backend: str | None = None,
    prefill_chunk: int | None = None,
    ssd_streaming: bool = False,
    mtp_model: str | None = None,
    vision: str | None = None,
    dspark: bool = False,
    mtp: bool = False,
) -> dict:
    """Estimate components in bytes, memoized per serve configuration.

    ds4's estimator runs in a subprocess (~0.3s measured against an 80.8 GiB
    GGUF, since it maps the GGUF to read the model shape), while `Model.memory()`
    runs per request;
    the memo makes a repeated ctx cheap and a new ctx pay one estimator run.
    """
    key = _signature(
        ds4_dir, gguf_path, ctx, binary, backend, prefill_chunk, ssd_streaming,
        mtp_model, vision, dspark, mtp,
    )
    cached = _ESTIMATE_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        result = run_estimator(
            ds4_dir=ds4_dir,
            gguf_path=gguf_path,
            ctx=ctx,
            binary=binary,
            backend=backend,
            prefill_chunk=prefill_chunk,
            ssd_streaming=ssd_streaming,
            mtp_model=mtp_model,
            vision=vision,
            dspark=dspark,
            mtp=mtp,
        )
        result["source"] = "ds4"
    except Ds4EstimatorError as exc:
        if key not in _WARNED:
            log.warning(
                f"ds4 VRAM estimate falling back to GGUF size + flat ctx term "
                f"(model={gguf_path} dir={ds4_dir}): {exc}"
            )
            _WARNED.add(key)
        result = fallback_estimate(
            ds4_dir=ds4_dir,
            gguf_path=gguf_path,
            ctx=ctx,
            mtp_model=mtp_model,
            vision=vision,
            dspark=dspark,
        )

    # Fallbacks are cached too: memory() runs per request, so re-running an
    # estimator that just failed would only slow the request down. _WARNED keeps
    # the log honest if the entry is later evicted and retried.
    _cache_put(key, result)
    return result


def estimate_mib(result: dict, sessions: int = 1) -> float:
    """Projected MiB from estimate components: GGUFs + N sessions of context.

    ds4 gives each resident session its own session graphs/caches (ds4-server
    logs the multiplied total) and its own drafter scratch, so `sessions` scales
    both per-session terms.
    """
    mib = 2**20
    sessions = max(int(sessions), 1)
    total_bytes = (
        result["model_bytes"]
        + result["support_bytes"]
        + result["vision_bytes"]
        + result["context_bytes"] * sessions
        + result.get("spec_graph_bytes", 0) * sessions
    )
    return total_bytes / mib + DS4_PROCESS_OVERHEAD_MIB
