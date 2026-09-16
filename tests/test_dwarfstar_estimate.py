"""ds4 VRAM footprint estimation (providers/dwarfstar_estimate.py).

The point of these tests is that the numbers a ds4 model is budgeted with come
from the ds4 build itself, are memoized per serve configuration, and degrade to
a size-aware fallback (with one loud warning) instead of a silent guess when the
estimator binary is absent or stale.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

import providers.dwarfstar_estimate as dse
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from providers.dwarfstar import DwarfStarProvider, warm_dwarfstar_estimates
from providers.lmstudio import LMStudioProvider

ESTIMATE = {
    "version": 3,
    "model_name": "DeepSeek V4 Flash",
    "backend": "metal",
    "ctx": 8192,
    "layers": 43,
    "model_bytes": 8_000_000_000,
    "support_bytes": 1_000_000_000,
    "vision_bytes": 0,
    "prefill_chunk": 4096,
    "prefill_cap": 4096,
    "raw_cap": 4352,
    "comp_cap": 2050,
    "raw_bytes": 100,
    "compressed_bytes": 200,
    "scratch_bytes": 300,
    "context_bytes": 1_000_000,
    # Per-session drafter scratch, as ds4 reports it (see ds4.h
    # ds4_spec_graph_memory): capture + verifier graph + host buffers.
    "dspark_capture_bytes": 200_000_000,
    "verifier_scratch_bytes": 80_000_000,
    "host_scratch_bytes": 1_000_000,
    "spec_graph_bytes": 281_000_000,
    "dspark_capture_stages": 3,
    "has_mtp": False,
    "mtp_draft_tokens": 0,
    # ds4's own shape identity (ds4_server.c server_model_id_from_engine) plus
    # the aliases its /v1/models lists for it (send_models()). YAALLB registers
    # models from these, so a Qwen GGUF never inherits DeepSeek's context.
    "model_family": "deepseek4",
    "model_id": 0,
    "model_aliases": ["deepseek-v4-flash", "deepseek-v4-pro"],
    # Some families (Qwen3.8, GLM) size their drafter graph elsewhere, so the
    # DeepSeek-path accessor pair does not describe them and the estimator says
    # so instead of printing DeepSeek-shaped numbers.
    "spec_graph_supported": True,
}

MIB = 2**20


@pytest.fixture(autouse=True)
def clean_estimate_caches():
    dse._ESTIMATE_CACHE.clear()
    dse._WARNED.clear()
    yield
    dse._ESTIMATE_CACHE.clear()
    dse._WARNED.clear()


class FakeEstimator:
    """Records every estimator invocation and replays a canned result."""

    def __init__(self, returncode=0, stdout=None, stderr="boom"):
        self.returncode = returncode
        self.stdout = json.dumps(ESTIMATE) if stdout is None else stdout
        self.stderr = stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))

        class Proc:
            pass

        proc = Proc()
        proc.returncode = self.returncode
        proc.stdout = self.stdout
        proc.stderr = self.stderr
        return proc

    @property
    def argvs(self):
        return [argv for argv, _ in self.calls]

    @property
    def kwargs(self):
        return [kw for _, kw in self.calls]


@pytest.fixture
def fake_estimator(monkeypatch):
    def install(**kwargs):
        fake = FakeEstimator(**kwargs)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


def test_run_estimator_passes_the_serve_configuration(fake_estimator):
    fake = fake_estimator()
    result = dse.run_estimator(
        ds4_dir="/tmp/ds4",
        gguf_path="./m.gguf",
        ctx=8192,
        backend="metal",
        prefill_chunk=2048,
        ssd_streaming=True,
        mtp_model="./support.gguf",
        vision="./vision.gguf",
        dspark=True,
    )

    # Paths are handed through untouched: the estimator runs with cwd=ds4_dir,
    # exactly like ds4-server, so a relative gguf_path means what config.json
    # says it means.
    assert fake.argvs[0] == [
        "./ds4-estimate",
        "--model",
        "./m.gguf",
        "--ctx",
        "8192",
        "--backend",
        "metal",
        "--prefill-chunk",
        "2048",
        "--ssd-streaming",
        "--mtp-model",
        "./support.gguf",
        "--dspark",
        "--vision",
        "./vision.gguf",
    ]
    assert fake.kwargs[0]["cwd"] == "/tmp/ds4"
    assert result["context_bytes"] == 1_000_000
    assert result["model_bytes"] == 8_000_000_000


def test_run_estimator_uses_a_private_ds4_instance_lock(fake_estimator):
    fake = fake_estimator()
    dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)
    dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=8192)

    # ds4 refuses to open a second engine on its instance lock, so each
    # estimator run needs its own (otherwise estimating next to a resident
    # ds4-server would fail outright).
    locks = {kw["env"]["DS4_LOCK_FILE"] for kw in fake.kwargs}
    assert len(locks) == 2
    assert all(lock.startswith(str(dse.DS4_LOCK_DIR)) for lock in locks)


def test_run_estimator_missing_binary_shows_the_build_command(monkeypatch):
    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(dse.Ds4EstimatorError) as exc:
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)
    # The hint is the command YAALLB itself runs: absolute -f, cwd = the tree.
    assert "make -f" in str(exc.value) and "tools/ds4-estimate.mk" in str(exc.value)
    assert "/tmp/ds4" in str(exc.value)


def test_run_estimator_reports_a_failing_estimator(fake_estimator):
    fake = fake_estimator(returncode=2, stdout="", stderr="cannot open model")
    with pytest.raises(dse.Ds4EstimatorError, match="cannot open model"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)


def test_run_estimator_passes_embedded_mtp(fake_estimator):
    # Qwen3.8 Flash Next and GLM 5.3 build their MTP drafter into the main GGUF
    # (--mtp, no --mtp-model file). Without the flag the estimator reports
    # mtp_draft_tokens=0 and budgets no drafter for a config that has one.
    fake = fake_estimator()
    dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=8192, mtp=True)
    assert "--mtp" in fake.argvs[0]
    dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="n.gguf", ctx=8192)
    assert "--mtp" not in fake.argvs[1]


def test_run_estimator_reports_the_model_identity(fake_estimator):
    fake_estimator()
    result = dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=8192)
    assert result["model_family"] == "deepseek4"
    assert result["model_id"] == 0
    assert result["model_aliases"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert result["spec_graph_supported"] is True


def test_run_estimator_rejects_a_schema_2_estimator(fake_estimator):
    # Schema 2 had no model_family/aliases, so it cannot say what model the
    # GGUF actually is: a Qwen GGUF would be served DeepSeek's model IDs and
    # context ceiling. A stale estimator must say "rebuild", not guess.
    stale = {
        key: value
        for key, value in ESTIMATE.items()
        if key
        not in (
            "model_family",
            "model_id",
            "model_aliases",
            "spec_graph_supported",
        )
    } | {"version": 2}
    fake_estimator(stdout=json.dumps(stale))
    with pytest.raises(dse.Ds4EstimatorError, match="rebuild"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    # Same schema but a helper that cannot state the identity: also a rebuild.
    nameless = {key: value for key, value in ESTIMATE.items() if key != "model_family"}
    fake_estimator(stdout=json.dumps(nameless))
    with pytest.raises(dse.Ds4EstimatorError, match="model_family"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    aliases = dict(ESTIMATE, model_aliases="deepseek-v4-flash")
    fake_estimator(stdout=json.dumps(aliases))
    with pytest.raises(dse.Ds4EstimatorError, match="model_aliases"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)


def test_run_estimator_rejects_a_foreign_schema(fake_estimator):
    stale = dict(ESTIMATE, version=99)
    fake_estimator(stdout=json.dumps(stale))
    with pytest.raises(dse.Ds4EstimatorError, match="rebuild"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    # A schema-1 helper (no drafter-scratch fields) predates ds4's
    # ds4_engine_spec_graph_memory_estimate(), so its numbers would silently
    # under-budget a drafter. It must say "rebuild", not fall back quietly.
    schema_one = {
        key: value
        for key, value in ESTIMATE.items()
        if key not in (
            "dspark_capture_bytes",
            "verifier_scratch_bytes",
            "host_scratch_bytes",
            "spec_graph_bytes",
            "dspark_capture_stages",
        )
    } | {"version": 1}
    fake_estimator(stdout=json.dumps(schema_one))
    with pytest.raises(dse.Ds4EstimatorError, match="rebuild"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    fake_estimator(stdout=json.dumps({**ESTIMATE, "spec_graph_bytes": None}))
    with pytest.raises(dse.Ds4EstimatorError, match="spec_graph_bytes"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    fake_estimator(stdout="ds4: model is not a GGUF file\n")
    with pytest.raises(dse.Ds4EstimatorError, match="rebuild"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)

    fake_estimator(stdout="null")
    with pytest.raises(dse.Ds4EstimatorError, match="not an object"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)


def test_fallback_uses_real_gguf_sizes_plus_a_flat_ctx_term(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * (3 * MIB))
    support = tmp_path / "support.gguf"
    support.write_bytes(b"y" * MIB)

    result = dse.fallback_estimate(
        ds4_dir=str(tmp_path),
        gguf_path="./m.gguf",
        ctx=8192,
        mtp_model=str(support),
    )

    assert result["source"] == "fallback"
    assert result["model_bytes"] == 3 * MIB
    assert result["support_bytes"] == MIB
    assert result["context_bytes"] == dse.DS4_CTX_BYTES_PER_TOKEN * 8192
    # A drafter's graph scratch cannot be read off file sizes, so the fallback
    # budgets its measured per-session stand-in rather than nothing.
    assert result["spec_graph_bytes"] == int(dse.DS4_DRAFTER_SCRATCH_FALLBACK_MIB * MIB)
    plain = dse.fallback_estimate(
        ds4_dir=str(tmp_path), gguf_path="./m.gguf", ctx=8192
    )
    assert plain["spec_graph_bytes"] == 0
    # A missing/renamed GGUF must not raise; it contributes nothing.
    assert dse.gguf_bytes(str(tmp_path), "./nope.gguf", None) == 0


def test_estimate_memoizes_one_serve_configuration(fake_estimator, monkeypatch):
    fake = fake_estimator()
    warnings = []
    monkeypatch.setattr(
        dse.log, "warning", lambda message: warnings.append(message)
    )

    kwargs = dict(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=8192)
    first = dse.estimate(**kwargs)
    second = dse.estimate(**kwargs)
    assert first is second
    assert len(fake.argvs) == 1
    assert first["source"] == "ds4"
    assert warnings == []

    # A different ctx (or a different drafter) is a different serve
    # configuration, so it costs exactly one more estimator run.
    dse.estimate(**kwargs)
    dse.estimate(**{**kwargs, "ctx": 16384})
    dse.estimate(**{**kwargs, "mtp_model": "./support.gguf"})
    # ... and so is turning on a drafter that lives inside the main GGUF.
    dse.estimate(**{**kwargs, "mtp": True})
    assert len(fake.argvs) == 4


def test_estimate_falls_back_once_and_warns(fake_estimator, monkeypatch):
    fake = fake_estimator(returncode=1, stdout="")
    warnings = []
    monkeypatch.setattr(
        dse.log, "warning", lambda message: warnings.append(message)
    )

    kwargs = dict(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)
    result = dse.estimate(**kwargs)
    assert result["source"] == "fallback"
    assert result["context_bytes"] == dse.DS4_CTX_BYTES_PER_TOKEN * 4096

    # The degraded estimate is cached like a good one: memory() runs per
    # request, and re-running a failing estimator there buys nothing.
    assert dse.estimate(**kwargs) is result
    assert len(fake.argvs) == 1

    # Even after the entry is evicted, the warning for that configuration is
    # not repeated on every request.
    dse._ESTIMATE_CACHE.clear()
    dse.estimate(**kwargs)
    assert len(fake.argvs) == 2
    assert len(warnings) == 1
    # The warning names both the degraded mode and the estimator's own failure,
    # so "no estimator binary" is distinguishable from "bad model path".
    assert "falling back" in warnings[0] and "exited 1" in warnings[0]


def test_estimate_mib_scales_the_per_session_terms_by_sessions():
    sessions = 4
    expected = (
        ESTIMATE["model_bytes"]
        + ESTIMATE["support_bytes"]
        + ESTIMATE["vision_bytes"]
        + ESTIMATE["context_bytes"] * sessions
        + ESTIMATE["spec_graph_bytes"] * sessions
    ) / MIB + dse.DS4_PROCESS_OVERHEAD_MIB
    assert dse.estimate_mib(ESTIMATE, sessions) == pytest.approx(expected)
    assert dse.estimate_mib(ESTIMATE, 0) == pytest.approx(
        dse.estimate_mib(ESTIMATE, 1)
    )


# --------------------------------------------------------------------------- #
# Provider integration
# --------------------------------------------------------------------------- #


def _provider(options=None, **extra):
    config = {"ds4_dir": "/tmp/ds4", "gguf_path": "./m.gguf", **extra}
    if options is not None:
        config["options"] = options
    return DwarfStarProvider(config=config)


def test_memory_projects_the_ds4_estimate(fake_estimator):
    fake = fake_estimator()
    provider = _provider()
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )

    # Weights + one session of context and drafter scratch + the fixed process
    # overhead, in MiB.
    assert model.memory() == pytest.approx(
        (8_000_000_000 + 1_000_000_000 + 1_000_000 + 281_000_000) / MIB
        + dse.DS4_PROCESS_OVERHEAD_MIB
    )

    # The estimator was told which ctx to price, with ds4_dir as its cwd.
    _, kwargs = fake.calls[0]
    assert kwargs["cwd"] == "/tmp/ds4"
    assert fake.argvs[0][-4:] == ["--model", "./m.gguf", "--ctx", "8192"]


def test_memory_counts_drafter_gguf_and_batched_sessions(fake_estimator):
    fake = fake_estimator()
    provider = _provider(
        options={
            "batched_session": 3,
            "mtp_model": "./support.gguf",
            "dspark": True,
            "metal": True,
        }
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )

    projected = model.memory()
    assert provider._sessions() == 3
    # Context and drafter scratch are per resident session (x3); the support
    # GGUF is mapped once.
    assert projected == pytest.approx(
        (8_000_000_000 + 1_000_000_000 + (1_000_000 + 281_000_000) * 3) / MIB
        + dse.DS4_PROCESS_OVERHEAD_MIB
    )
    assert projected > dse.estimate_mib(ESTIMATE, 1)
    assert fake.argvs[0] == [
        "./ds4-estimate",
        "--model",
        "./m.gguf",
        "--ctx",
        "8192",
        "--backend",
        "metal",
        "--mtp-model",
        "./support.gguf",
        "--dspark",
    ]


def test_backend_name_follows_the_configured_flags():
    assert _provider().options == {}
    assert _provider()._backend_name() is None
    assert _provider(options={"metal": True})._backend_name() == "metal"
    assert _provider(options={"cpu": True})._backend_name() == "cpu"
    # An explicit --backend wins over a bare backend-selecting flag.
    assert (
        _provider(options={"backend": "cuda", "metal": True})._backend_name() == "cuda"
    )


def test_sessions_default_to_one_and_ignore_nonsense():
    assert _provider()._sessions() == 1
    assert _provider(options={"batched_session": 2})._sessions() == 2
    assert _provider(options={"batched_session": "lots"})._sessions() == 1
    assert _provider(options={"batched_session": 0})._sessions() == 1


def test_warm_dwarfstar_estimates_covers_configured_contexts(fake_estimator, monkeypatch):
    fake = fake_estimator()
    messages = []
    monkeypatch.setattr(dse.log, "info", lambda message: messages.append(message))

    provider = _provider(ctx_length=100000)
    warm_dwarfstar_estimates([provider], default_ctx=4096)

    # Provider ctx plus the router default, cheapest first, one run each.
    assert [argv[-1] for argv in fake.argvs] == ["4096", "100000"]
    assert len(messages) == 2
    assert "ctx=4096" in messages[0] and "source=ds4" in messages[0]


class FakeMake:
    """Records make invocations and replays canned exit codes per attempt."""

    def __init__(self, returncodes=(0,), stdout="", stderr="cc: nope"):
        self.returncodes = list(returncodes)
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))

        class Proc:
            pass

        proc = Proc()
        proc.returncode = self.returncodes[min(len(self.calls) - 1, len(self.returncodes) - 1)]
        proc.stdout = self.stdout
        proc.stderr = self.stderr
        return proc

    @property
    def argvs(self):
        return [argv for argv, _ in self.calls]

    @property
    def kwargs(self):
        return [kw for _, kw in self.calls]


@pytest.fixture
def fake_make(monkeypatch):
    def install(**kwargs):
        fake = FakeMake(**kwargs)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


@pytest.fixture
def ds4_tree(tmp_path):
    """A ds4 tree that looks built: a Makefile plus one core object."""
    root = tmp_path / "ds4"
    root.mkdir()
    (root / "Makefile").write_text("all:\n\techo built\n")
    (root / "ds4.o").write_bytes(b"o")
    return root


def test_build_estimator_runs_make_inside_the_ds4_tree(fake_make, ds4_tree):
    fake = fake_make()
    assert dse.build_estimator(str(ds4_tree)) == str(ds4_tree)

    argv, kwargs = fake.argvs[0], fake.kwargs[0]
    # Absolute -f and cwd = the tree: make -C would make the fragment's own
    # relative paths (its `include Makefile`, ds4's object rules) ambiguous.
    assert argv[:2] == ["make", "-f"]
    assert os.path.isabs(argv[2]) and argv[2].endswith("tools/ds4-estimate.mk")
    assert os.path.exists(argv[2])
    assert "-C" not in argv
    assert kwargs["cwd"] == os.path.abspath(str(ds4_tree))


def test_build_estimator_retries_cpu_only_then_reports_both(fake_make, ds4_tree):
    # A tree built with `make cpu` has no GPU objects to link against, so the
    # CPU-only variant is tried before giving up.
    fake = fake_make(returncodes=(1, 1), stderr="Undefined symbols: _metal_graph_alloc")
    with pytest.raises(dse.Ds4BuildError) as exc:
        dse.build_estimator(str(ds4_tree))
    assert len(fake.argvs) == 2
    assert fake.argvs[1][-1] == "DS4_ESTIMATE_CPU_ONLY=1"
    assert "Undefined symbols" in str(exc.value)
    assert "make -f" in str(exc.value)

    # A tree whose CPU-only variant links is a success, not an error.
    ok = fake_make(returncodes=(1, 0))
    assert dse.build_estimator(str(ds4_tree)) == str(ds4_tree)
    assert len(ok.argvs) == 2
    assert ok.argvs[1][-1] == "DS4_ESTIMATE_CPU_ONLY=1"


def test_build_estimator_refuses_to_compile_the_whole_engine(fake_make, tmp_path):
    unbuilt = tmp_path / "ds4-src"
    unbuilt.mkdir()
    (unbuilt / "Makefile").write_text("all:\n\tcc -c ds4.c\n")
    (unbuilt / "ds4.c").write_text("int main(void){return 0;}\n")

    fake = fake_make()
    with pytest.raises(dse.Ds4BuildError, match="build ds4"):
        dse.build_estimator(str(unbuilt))
    # YAALLB appends its own helper to a built engine; it does not decide to
    # spend twenty minutes compiling somebody's inference engine instead.
    assert fake.argvs == []

    with pytest.raises(dse.Ds4BuildError, match="not a directory"):
        dse.build_estimator(str(tmp_path / "not-a-tree"))

    wrong = tmp_path / "not-ds4"
    wrong.mkdir()
    with pytest.raises(dse.Ds4BuildError, match="Makefile"):
        dse.build_estimator(str(wrong))


def test_build_estimator_says_when_make_is_missing(monkeypatch, ds4_tree):
    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(dse.Ds4BuildError, match="make"):
        dse.build_estimator(str(ds4_tree))


def test_build_dwarfstar_estimators_builds_each_tree_once(fake_make, tmp_path, ds4_tree):
    from providers.dwarfstar import build_dwarfstar_estimators

    other = tmp_path / "other-ds4"
    other.mkdir()
    (other / "Makefile").write_text("all:\n\techo built\n")
    (other / "ds4_cpu.o").write_bytes(b"o")

    fake = fake_make()
    providers = [
        DwarfStarProvider(0, {"ds4_dir": str(ds4_tree), "gguf_path": "a.gguf"}),
        DwarfStarProvider(1, {"ds4_dir": str(ds4_tree) + "/", "gguf_path": "b.gguf"}),
        DwarfStarProvider(2, {"ds4_dir": str(other), "gguf_path": "c.gguf"}),
    ]
    build_dwarfstar_estimators(providers + [LMStudioProvider()])

    # Two instances of one tree means one build; a second tree gets its own.
    dirs = [kw["cwd"] for kw in fake.kwargs]
    assert dirs == [os.path.abspath(str(ds4_tree)), os.path.abspath(str(other))]


def test_build_dwarfstar_estimators_needs_a_ds4_dir(fake_make):
    from providers.dwarfstar import build_dwarfstar_estimators

    fake = fake_make()
    with pytest.raises(dse.Ds4BuildError, match="ds4_dir"):
        build_dwarfstar_estimators([DwarfStarProvider(0, {})])
    assert fake.argvs == []


def test_build_estimator_is_a_real_make_run(tmp_path):
    # No mocks: the argv/cwd contract has to survive an actual make, including
    # the fragment-style `include Makefile` the real one relies on.
    root = tmp_path / "tree"
    root.mkdir()
    (root / "Makefile").write_text("CORE_OBJS =\nCPU_CORE_OBJS =\nall:\n\techo engine\n")
    (root / "ds4.o").write_bytes(b"o")
    fragment = tmp_path / "estimate.mk"
    fragment.write_text(
        "include Makefile\n"
        ".DEFAULT_GOAL := ds4-estimate\n"
        "ds4-estimate:\n"
        "\tprintf '#!/bin/sh\\nexit 0\\n' > $@\n"
        "\tchmod +x $@\n"
    )

    proc = subprocess.run(
        ["make", "-f", str(fragment)], cwd=root, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    built = root / "ds4-estimate"
    assert built.exists() and os.access(built, os.X_OK)


def test_estimate_mk_offers_a_cpu_only_variant():
    makefile = (Path(__file__).resolve().parent.parent / "tools" / "ds4-estimate.mk").read_text()
    assert "DS4_ESTIMATE_CPU_ONLY" in makefile
    # The CPU switch must swap in the CPU object list itself: passing
    # CORE_OBJS="$(CPU_CORE_OBJS)" on a make command line is not portable.
    assert "CORE_OBJS := $(CPU_CORE_OBJS)" in makefile


def test_warm_dwarfstar_estimates_without_a_configured_gguf(monkeypatch):
    provider = DwarfStarProvider(config={"ds4_dir": "/tmp/ds4"})
    messages = []
    monkeypatch.setattr(dse.log, "warning", lambda message: messages.append(message))

    # Warns about the unconfigured provider instead of raising on it.
    warm_dwarfstar_estimates([provider], default_ctx=4096)

    assert len(messages) == 1
    assert "gguf_path" in messages[0]
