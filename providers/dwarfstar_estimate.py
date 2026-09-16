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


# Where the estimator's make fragment lives. It is handed to make as an
# absolute path, with the ds4 tree as the working directory, so the fragment's
# own relative paths (`include Makefile`, ds4's object rules) unambiguously mean
# that tree. `make -C` invites exactly that confusion, so it is not used.
DS4_ESTIMATE_MAKEFILE = (
    Path(__file__).resolve().parent.parent / "tools" / "ds4-estimate.mk"
)

# A ds4 tree that has never been compiled has neither of these. Compiling an
# inference engine is not YAALLB's job (see build_estimator), but noticing that
# it would have to is.
DS4_CORE_OBJECTS = ("ds4.o", "ds4_cpu.o")

# make may relink the estimator, or recompile one stale core object it links.
# It is never asked to build an engine from scratch, so this is generous.
DS4_BUILD_TIMEOUT = 1800


class Ds4BuildError(RuntimeError):
    """The ds4 estimator could not be built into a ds4 tree."""


def build_hint(ds4_dir: str) -> str:
    """The command that produces the estimator binary for a ds4 tree.

    It is the command YAALLB itself runs: absolute `-f`, the tree as the working
    directory.
    """
    return f"(cd {os.path.abspath(ds4_dir)} && make -f {DS4_ESTIMATE_MAKEFILE})"


def _make_command(cpu_only: bool) -> list[str]:
    argv = ["make", "-f", str(DS4_ESTIMATE_MAKEFILE)]
    if cpu_only:
        # A tree built with `make cpu` has no GPU objects to link against.
        argv.append("DS4_ESTIMATE_CPU_ONLY=1")
    return argv


def build_estimator(ds4_dir: str) -> str:
    """Build the estimator into one ds4 tree and return that tree's abs path.

    The estimator is YAALLB's own program linked against the tree's *already
    compiled* objects, because only that build knows its own footprint
    arithmetic. Building it therefore adds two files to the tree (`ds4-estimate`
    and `ds4_estimate.host.o`) and changes nothing else - ds4's sources are
    never touched - and make does no work at all when the tree is current.

    Raises Ds4BuildError when the tree is not a buildable ds4 tree, when `make`
    is unavailable, or when the build itself fails (a `make cpu` tree fails the
    normal link, so the CPU-only variant is retried once and both attempts are
    reported).
    """
    root = os.path.abspath(ds4_dir)
    if not os.path.isdir(root):
        raise Ds4BuildError(
            f"cannot build the ds4 estimator: ds4_dir {ds4_dir!r} is not a directory"
        )
    if not os.path.isfile(os.path.join(root, "Makefile")):
        raise Ds4BuildError(
            f"cannot build the ds4 estimator: {root} has no Makefile; ds4_dir "
            "must point at a ds4 source tree"
        )
    if not any(
        os.path.exists(os.path.join(root, obj)) for obj in DS4_CORE_OBJECTS
    ):
        raise Ds4BuildError(
            f"{root} has no compiled ds4 objects (looked for "
            f"{' or '.join(DS4_CORE_OBJECTS)}). YAALLB appends its estimator to "
            "a built engine rather than compiling one; build ds4 first with "
            f"(cd {root} && make)."
        )

    failures = []
    for cpu_only in (False, True):
        argv = _make_command(cpu_only)
        try:
            proc = subprocess.run(
                argv,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=DS4_BUILD_TIMEOUT,
            )
        except FileNotFoundError as exc:
            raise Ds4BuildError(
                f"make is not available, so the ds4 estimator cannot be built "
                f"in {root}; build it by hand: {build_hint(root)}"
            ) from exc
        except subprocess.TimeoutExpired:
            failures.append(f"{' '.join(argv)} timed out after {DS4_BUILD_TIMEOUT}s")
            continue
        if proc.returncode == 0:
            return root
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-10:])
        failures.append(f"{' '.join(argv)} exited {proc.returncode}:\n{tail}")

    raise Ds4BuildError(
        f"building the ds4 estimator failed in {root}:\n\n"
        + "\n\n".join(failures)
        + f"\n\nRun it by hand to see the whole error: {build_hint(root)}"
    )


def env_value(value) -> str:
    """A ds4 environment value as ds4 reads it: strings, and 1/0 for booleans."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError(f"ds4 env values must be scalars, got {type(value).__name__}")


def merged_env(extra: dict | None) -> dict | None:
    """The environment for a child process, or None to inherit unchanged.

    None (nothing configured) keeps subprocess's inherit semantics instead of
    snapshotting os.environ early.
    """
    if not extra:
        return None
    return {**os.environ, **{key: env_value(value) for key, value in extra.items()}}


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
    extra_env: dict | None = None,
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
    # The environment ds4-server will be started with, so the shape being priced
    # is the shape that gets served (DS4_QWEN4_YARN_FACTOR changes what a context
    # costs, among others).
    env = merged_env(extra_env) or dict(os.environ)
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


# Families with a measured flat context term, in bytes per context token,
# including the transients that term has always bundled in. DeepSeek V4 Flash on
# Metal is the shape ``DS4_CTX_BYTES_PER_TOKEN`` was measured from (16,416 against
# 16,801 measured at ctx 1,000,000 today).
#
# Other families are deliberately absent: Qwen3.8 Flash Next's footprint is not
# one straight line - its KV is 34,128 bytes/token, but its transients follow
# --prefill-chunk (1,298 MiB at a 1024 chunk vs 7,928 MiB at its default 8192, for
# the same 32,768-token context) and include a multi-GiB fixed term. Guessing a
# DeepSeek slope there would under-budget the model by gigabytes, which is the
# one thing a VRAM scheduler must not do, so it refuses instead (see
# ``fallback_estimate``).
DS4_FLAT_CTX_BYTES_PER_TOKEN = {"deepseek4": DS4_CTX_BYTES_PER_TOKEN}


def fallback_estimate(
    *,
    ds4_dir: str,
    gguf_path: str,
    ctx: int,
    mtp_model: str | None = None,
    vision: str | None = None,
    dspark: bool = False,
    model_family: str | None = None,
) -> dict:
    """Size-aware estimate for when the ds4 estimator cannot run.

    Uses the real GGUF bytes (the dominant term) plus a flat context term, and a
    per-session drafter stand-in (``DS4_DRAFTER_SCRATCH_FALLBACK_MIB``) when one
    is configured, since none of that scratch follows from file sizes.

    A flat context term only exists for families measured that way. With no
    family known, this is the estimate YAALLB has always produced; with a family
    known that has no flat model (Qwen3.8, GLM), it refuses rather than
    understating a model whose transients follow other options - the estimator is
    built automatically now, so reaching this branch means that build needs
    fixing.
    """
    if model_family is None:
        flat = DS4_CTX_BYTES_PER_TOKEN
    else:
        flat = DS4_FLAT_CTX_BYTES_PER_TOKEN.get(model_family)
        if flat is None:
            raise Ds4EstimatorError(
                f"no flat VRAM model for ds4 family {model_family!r}: its footprint "
                "does not follow one bytes-per-token slope (Qwen3.8 Flash Next's "
                "transients follow --prefill-chunk), and YAALLB will not budget it "
                "from another family's numbers. Fix the estimator build "
                f"({build_hint(ds4_dir)}) so ds4 can be asked directly."
            )
    context_bytes = flat * max(int(ctx), 1)
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
    mtp_model, vision, dspark, mtp, model_family, extra_env,
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
        # A configured family changes what the fallback is allowed to guess,
        # so it is part of the configuration, not just of the call.
        model_family,
        # ds4's own env knobs change the answer, so they are part of the
        # configuration being memoized.
        tuple(sorted((extra_env or {}).items())),
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
    model_family: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Estimate components in bytes, memoized per serve configuration.

    ds4's estimator runs in a subprocess (~0.3s measured against an 80.8 GiB
    GGUF, since it maps the GGUF to read the model shape), while `Model.memory()`
    runs per request;
    the memo makes a repeated ctx cheap and a new ctx pay one estimator run.
    """
    key = _signature(
        ds4_dir, gguf_path, ctx, binary, backend, prefill_chunk, ssd_streaming,
        mtp_model, vision, dspark, mtp, model_family, extra_env,
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
            extra_env=extra_env,
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
            # Only a family named in config is known here: an estimate that
            # failed carries no shape of its own.
            model_family=model_family,
        )

    # Fallbacks are cached too: memory() runs per request, so re-running an
    # estimator that just failed would only slow the request down. _WARNED keeps
    # the log honest if the entry is later evicted and retried.
    _cache_put(key, result)
    return result


def _shared_prefill_workspace(result: dict, sessions: int) -> bool:
    """Whether ds4 aliases one prefill arena across the batched sessions.

    ds4-server turns the shared workspace on whenever it batches sessions
    (`share_session_prefill_workspace = batched_sessions > 0`) and its Metal
    session graphs alias it (`share = e->share_session_prefill_workspace &&
    e->backend == DS4_BACKEND_METAL`, in both the DeepSeek and the Qwen3.8
    session paths), so the chunk-sized transients are allocated once. Other
    backends keep private graphs. docs/SERVER.md: "an extra slot costs its caches
    rather than another few GiB of transients".
    """
    return sessions > 1 and result.get("backend") == "metal" and "scratch_bytes" in result


def estimate_mib(result: dict, sessions: int = 1) -> float:
    """Projected MiB from estimate components: GGUFs + N sessions of context.

    ds4 gives each resident session its own caches and its own drafter scratch
    (ds4-server logs the multiplied total), so those scale with `sessions`. The
    chunk-sized transients scale only where ds4 does not share them: for Qwen3.8
    Flash Next they are the largest part of the context term by far (7,928 MiB of
    a 9,000 MiB context at the default prefill chunk), so counting them per
    batched session would over-evict for a saving ds4 does not need.
    """
    mib = 2**20
    sessions = max(int(sessions), 1)
    if _shared_prefill_workspace(result, sessions):
        context = (
            (result["raw_bytes"] + result["compressed_bytes"]) * sessions
            + result["scratch_bytes"]
        )
    else:
        context = result["context_bytes"] * sessions
    total_bytes = (
        result["model_bytes"]
        + result["support_bytes"]
        + result["vision_bytes"]
        + context
        + result.get("spec_graph_bytes", 0) * sessions
    )
    return total_bytes / mib + DS4_PROCESS_OVERHEAD_MIB
