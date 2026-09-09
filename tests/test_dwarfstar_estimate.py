"""ds4 VRAM footprint estimation (providers/dwarfstar_estimate.py).

The point of these tests is that the numbers a ds4 model is budgeted with come
from the ds4 build itself, are memoized per serve configuration, and degrade to
a size-aware fallback (with one loud warning) instead of a silent guess when the
estimator binary is absent or stale.
"""

import json
import subprocess

import pytest

import providers.dwarfstar_estimate as dse
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from providers.dwarfstar import DwarfStarProvider, warm_dwarfstar_estimates

ESTIMATE = {
    "version": 1,
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
    "has_mtp": False,
    "mtp_draft_tokens": 0,
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
    assert "make -C /tmp/ds4" in str(exc.value)
    assert "tools/ds4-estimate.mk" in str(exc.value)


def test_run_estimator_reports_a_failing_estimator(fake_estimator):
    fake = fake_estimator(returncode=2, stdout="", stderr="cannot open model")
    with pytest.raises(dse.Ds4EstimatorError, match="cannot open model"):
        dse.run_estimator(ds4_dir="/tmp/ds4", gguf_path="m.gguf", ctx=4096)


def test_run_estimator_rejects_a_foreign_schema(fake_estimator):
    stale = dict(ESTIMATE, version=99)
    fake_estimator(stdout=json.dumps(stale))
    with pytest.raises(dse.Ds4EstimatorError, match="rebuild"):
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
    assert len(fake.argvs) == 3


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


def test_estimate_mib_scales_the_context_term_by_sessions():
    sessions = 4
    expected = (
        ESTIMATE["model_bytes"]
        + ESTIMATE["support_bytes"]
        + ESTIMATE["vision_bytes"]
        + ESTIMATE["context_bytes"] * sessions
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

    # Weights + one session of context + the fixed process overhead, in MiB.
    assert model.memory() == pytest.approx(
        (8_000_000_000 + 1_000_000_000 + 1_000_000) / MIB
        + dse.DS4_PROCESS_OVERHEAD_MIB
    )

    # The estimator was told which ctx to price, with ds4_dir as its cwd.
    _, kwargs = fake.calls[0]
    assert kwargs["cwd"] == "/tmp/ds4"
    assert fake.argvs[0][-4:] == ["--model", "./m.gguf", "--ctx", "8192"]


def test_memory_counts_drafter_gguf_and_batched_sessions(fake_estimator):
    fake = fake_estimator()
    provider = _provider(
        options={"batched_session": 3, "mtp_model": "./support.gguf", "metal": True}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )

    projected = model.memory()
    assert provider._sessions() == 3
    # Context is per resident session (x3); the support GGUF is mapped once.
    assert projected == pytest.approx(
        (8_000_000_000 + 1_000_000_000 + 1_000_000 * 3) / MIB
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


def test_warm_dwarfstar_estimates_without_a_configured_gguf(monkeypatch):
    provider = DwarfStarProvider(config={"ds4_dir": "/tmp/ds4"})
    messages = []
    monkeypatch.setattr(dse.log, "warning", lambda message: messages.append(message))

    # Warns about the unconfigured provider instead of raising on it.
    warm_dwarfstar_estimates([provider], default_ctx=4096)

    assert len(messages) == 1
    assert "gguf_path" in messages[0]
