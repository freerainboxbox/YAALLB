import pytest

import httpx

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model
from abstractions.provider import Provider
from providers.dwarfstar import DS4_CONTEXT_LENGTH, DwarfStarProvider
from providers.lmstudio import LMStudioProvider


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        Provider()


def test_model_is_abstract():
    with pytest.raises(TypeError):
        Model(ModelDescriptor("m", LMStudioProvider()), LoadOptions())


def test_provider_model_is_mandatory_subclass():
    assert issubclass(LMStudioProvider.Model, Model)
    assert issubclass(DwarfStarProvider.Model, Model)


def test_descriptor_holds_model_id_and_provider():
    provider = LMStudioProvider()
    desc = ModelDescriptor("model-x", provider)
    assert desc.modelId == "model-x"
    assert desc.provider is provider


def test_load_options_default_and_extra_attrs():
    opts = LoadOptions()
    assert opts.ctx_length == 4096
    opts2 = LoadOptions(ctx_length=8192, top_p=0.9, temperature=0.2)
    assert opts2.ctx_length == 8192
    assert opts2.top_p == 0.9
    assert opts2.temperature == 0.2


def test_create_model_returns_provider_model():
    provider = LMStudioProvider()
    desc = ModelDescriptor("m", provider)
    model = provider.createModel(desc, LoadOptions())
    assert isinstance(model, provider.Model)
    assert model.descriptor is desc
    assert model.loadOptions is not None


def test_load_and_unload_alias_to_provider(monkeypatch):
    def fake_post(url, json, headers, timeout=None):
        class Resp:
            status_code = 200
            text = "ok"

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)

    provider = LMStudioProvider()
    desc = ModelDescriptor("m", provider)
    model = provider.createModel(desc, LoadOptions())
    assert not model.loaded
    model.load()
    assert model.loaded
    model.unloadModel()
    assert not model.loaded


def test_providers_concrete_memory(monkeypatch):
    import subprocess

    def fake_run(argv, **kw):
        class Proc:
            returncode = 0
            stdout = "Estimated GPU Memory:   20.39 GiB"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p1 = LMStudioProvider()
    m1 = p1.createModel(ModelDescriptor("a", p1), LoadOptions())
    assert isinstance(m1.memory(), float)
    p2 = DwarfStarProvider()
    m2 = p2.createModel(ModelDescriptor("b", p2), LoadOptions())
    assert isinstance(m2.memory(), float)


def test_providers_list_descriptors(monkeypatch):
    def fake_get(url, headers):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"models": []}

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.get", fake_get)
    assert LMStudioProvider().getModelsDescriptors() == []
    provider = DwarfStarProvider()
    descs = provider.getModelsDescriptors()
    assert [d.modelId for d in descs] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert all(d.provider is provider for d in descs)


def test_dwarfstar_memory_piecewise():
    # Without a provider ctx_length, memory uses the resident model's ctx
    # (or DS4_CONTEXT_LENGTH when nothing is resident). With one set, the
    # provider-level ctx wins regardless of the model's load options.
    p = DwarfStarProvider()
    small = p.createModel(ModelDescriptor("b", p), LoadOptions(ctx_length=4096))
    assert small.memory() == pytest.approx(83065.32 + 16416 * DS4_CONTEXT_LENGTH / (2**20))

    p2 = DwarfStarProvider(config={"ctx_length": 8192})
    big = p2.createModel(ModelDescriptor("b", p2), LoadOptions(ctx_length=4096))
    assert big.memory() == pytest.approx(83065.32 + 16416 * 8192 / (2**20))


def test_lmstudio_memory_parses_gib(monkeypatch):
    import subprocess
    from providers.lmstudio import LMStudioProvider

    calls = {}

    def fake_run(argv, **kw):
        calls["argv"] = argv
        class Proc:
            returncode = 0
            stdout = ""
            stderr = (
                "Model: google/gemma-4-26b-a4b-qat\n"
                "Context Length: 100,000\n"
                "Estimated GPU Memory:   20.39 GiB\n"
                "Estimated Total Memory: 20.39 GiB\n"
            )

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LMStudioProvider()
    m = p.createModel(
        ModelDescriptor("google/gemma-4-26b-a4b-qat", p), LoadOptions(ctx_length=100000)
    )
    assert m.memory() == pytest.approx(20.39 * 1024)
    assert calls["argv"] == [
        "lms",
        "load",
        "-y",
        "--estimate-only",
        "-c",
        "100000",
        "google/gemma-4-26b-a4b-qat",
    ]


def test_lmstudio_memory_parses_mib(monkeypatch):
    import subprocess
    from providers.lmstudio import LMStudioProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 0
            stdout = ""
            stderr = "Estimated GPU Memory:   444.28 MiB"

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("qwen3-0.6b-mlx", p), LoadOptions(ctx_length=100000))
    assert m.memory() == pytest.approx(444.28)


def test_lmstudio_memory_raises_on_nonzero_exit(monkeypatch):
    import subprocess
    from providers.lmstudio import LMStudioProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 1
            stdout = ""
            stderr = "boom"

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("nope", p), LoadOptions())
    with pytest.raises(RuntimeError):
        m.memory()


def test_lmstudio_get_models_descriptors_filters_llm(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    captured = {}

    def fake_get(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "models": [
                        {"key": "a/llm", "type": "llm"},
                        {"key": "b/emb", "type": "embedding"},
                        {"key": "c/llm", "type": "llm"},
                    ]
                }

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.get", fake_get)

    p = LMStudioProvider(config={"api_key": "sk"})
    descs = p.getModelsDescriptors()
    assert [d.modelId for d in descs] == ["a/llm", "c/llm"]
    assert all(d.provider is p for d in descs)
    assert captured["url"] == "http://127.0.0.1:1234/api/v1/models"
    assert captured["headers"] == {"Authorization": "Bearer sk"}


def test_lmstudio_load_calls_rest_api(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    calls = []

    def fake_post(url, json, headers, timeout=None):
        calls.append((url, json, headers))
        class Resp:
            status_code = 200
            text = "ok"

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)

    p = LMStudioProvider(config={"api_key": "sk"})
    m = p.createModel(
        ModelDescriptor("qwen3-0.6b-mlx", p), LoadOptions(ctx_length=8192)
    )
    p.loadModel(m)

    assert m.loaded
    assert calls[0][0] == "http://127.0.0.1:1234/api/v1/models/load"
    assert calls[0][1] == {
        "model": "qwen3-0.6b-mlx",
        "context_length": 8192,
    }
    assert calls[0][2] == {"Authorization": "Bearer sk"}


def test_lmstudio_unload_calls_rest_api(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    calls = []

    def fake_post(url, json, headers, timeout=None):
        calls.append((url, json, headers))
        class Resp:
            status_code = 200
            text = "ok"

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)

    p = LMStudioProvider(config={"api_key": "sk"})
    m = p.createModel(
        ModelDescriptor("qwen3-0.6b-mlx", p), LoadOptions(ctx_length=8192)
    )
    p.loadModel(m)
    p.unloadModel(m)

    assert not m.loaded
    assert calls[1][0] == "http://127.0.0.1:1234/api/v1/models/unload"
    assert calls[1][1] == {"instance_id": "qwen3-0.6b-mlx"}
    assert calls[1][2] == {"Authorization": "Bearer sk"}


def test_lmstudio_load_raises_on_error_status(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    def fake_post(url, json, headers, timeout=None):
        class Resp:
            status_code = 500
            text = "boom"

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("x", p), LoadOptions())
    with pytest.raises(RuntimeError):
        p.loadModel(m)
    assert not m.loaded


def test_lmstudio_memory_raises_on_unparseable(monkeypatch):
    import subprocess
    from providers.lmstudio import LMStudioProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 0
            stdout = "unexpected output"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("nope", p), LoadOptions())
    with pytest.raises(RuntimeError):
        m.memory()


def test_lmstudio_load_timeout_then_poll_loads(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    def fake_post(url, json, headers, timeout=None):
        raise httpx.ReadTimeout("timed out", request=None)

    calls = []

    def fake_get(url, headers=None):
        calls.append(url)
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"key": "qwen3-0.6b-mlx", "type": "llm"}]}

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)
    monkeypatch.setattr("providers.lmstudio.httpx.get", fake_get)

    p = LMStudioProvider()
    m = p.createModel(
        ModelDescriptor("qwen3-0.6b-mlx", p), LoadOptions(ctx_length=8192)
    )
    p.loadModel(m)

    assert m.loaded
    assert calls == ["http://127.0.0.1:1234/api/v1/models"]


def test_lmstudio_load_still_loading_4xx_then_poll(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    def fake_post(url, json, headers, timeout=None):
        class Resp:
            status_code = 400
            text = "still loading"

        return Resp()

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"key": "x", "type": "llm"}]}

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)
    monkeypatch.setattr("providers.lmstudio.httpx.get", fake_get)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("x", p), LoadOptions())
    p.loadModel(m)

    assert m.loaded


def test_lmstudio_load_never_loads_raises(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    def fake_post(url, json, headers, timeout=None):
        raise httpx.ReadTimeout("timed out", request=None)

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": []}

        return Resp()

    monkeypatch.setattr("providers.lmstudio.httpx.post", fake_post)
    monkeypatch.setattr("providers.lmstudio.httpx.get", fake_get)
    monkeypatch.setattr("providers.lmstudio.time.sleep", lambda *a: None)
    monkeypatch.setattr("providers.lmstudio.LMS_LOAD_MAX_WAIT", 0.01)

    p = LMStudioProvider()
    m = p.createModel(ModelDescriptor("nope", p), LoadOptions())
    with pytest.raises(RuntimeError):
        p.loadModel(m)
    assert not m.loaded
