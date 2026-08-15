import pytest

import httpx

from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model
from abstractions.provider import Provider
from providers.dwarfstar import DS4_CONTEXT_LENGTH, DwarfStarProvider
from providers.llama_cpp import LlamaCppProvider
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
    assert issubclass(LlamaCppProvider.Model, Model)


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


# ---- llama_cpp memory estimation ----


def test_llama_cpp_provider_type_id_and_single_resident():
    assert LlamaCppProvider()._type_id == "llama_cpp"
    assert LlamaCppProvider().single_resident


def test_llama_cpp_defaults():
    p = LlamaCppProvider()
    assert p.host == "127.0.0.1"
    assert p.port == 8080
    assert p.endpoint_uri == "http://127.0.0.1:8080/v1"
    assert p.llama_cpp_dir is None
    assert p.gguf_path is None
    assert p.ctx_length is None
    assert p.alias is None


def test_llama_cpp_memory_parses_fit_params(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    calls = {}

    def fake_run(argv, **kw):
        calls["argv"] = argv
        calls["kwargs"] = kw
        class Proc:
            returncode = 0
            stdout = "MTL0 35905 142 493 \nHost 515 0 24 \n"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/path/to/llama-cpp",
            "gguf_path": "model.gguf",
        }
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions(ctx_length=4096))
    # VRAM footprint is the GPU (non-Host) device total: 35905+142+493.
    assert m.memory() == pytest.approx(36540.0)
    assert calls["kwargs"]["cwd"] == "/path/to/llama-cpp"
    assert calls["argv"] == [
        "/path/to/llama-cpp/llama-fit-params",
        "-m",
        "model.gguf",
        "--ctx-size",
        "4096",
        "--fit-print",
        "on",
    ]


def test_llama_cpp_memory_forwards_core_options(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    argv = {}

    def fake_run(argv_actual, **kw):
        argv["actual"] = argv_actual
        class Proc:
            returncode = 0
            stdout = "MTL0 100 200 300 \n"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/path",
            "gguf_path": "m.gguf",
            "ctx_length": 8192,
            "options": {"ngl": 24, "cache_type_k": "q8_0", "lora": "lora.bin"},
        }
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions(ctx_length=4096))
    assert m.memory() == pytest.approx(600.0)
    # Core options ride into the llama-fit-params estimate so VRAM reflects them.
    assert argv["actual"] == [
        "/path/llama-fit-params",
        "-m",
        "m.gguf",
        "--ctx-size",
        "8192",
        "--fit-print",
        "on",
        "--gpu-layers",
        "24",
        "--cache-type-k",
        "q8_0",
        "--lora",
        "lora.bin",
    ]


def test_llama_cpp_memory_provider_ctx_overrides_model(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    argv = {}

    def fake_run(argv_actual, **kw):
        argv["actual"] = argv_actual
        class Proc:
            returncode = 0
            stdout = "MTL0 100 200 300 \n"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/path",
            "gguf_path": "m.gguf",
            "ctx_length": 8192,
        }
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions(ctx_length=4096))
    # Provider-level ctx_length wins over the model's load options.
    assert m.memory() == pytest.approx(600.0)
    assert argv["actual"][-4:] == ["--ctx-size", "8192", "--fit-print", "on"]


def test_llama_cpp_memory_raises_on_nonzero_exit(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 1
            stdout = ""
            stderr = "boom"

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={"llama_cpp_dir": "/path", "gguf_path": "m.gguf"}
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions())
    with pytest.raises(RuntimeError):
        m.memory()


def test_llama_cpp_memory_raises_on_unparseable(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 0
            stdout = "unexpected output"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={"llama_cpp_dir": "/path", "gguf_path": "m.gguf"}
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions())
    with pytest.raises(RuntimeError):
        m.memory()


def test_llama_cpp_memory_requires_dir_and_gguf():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider()
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions())
    with pytest.raises(ValueError):
        m.memory()


def test_llama_cpp_memory_raises_runtime_error_on_non_numeric(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    def fake_run(argv, **kw):
        class Proc:
            returncode = 0
            stdout = "MTL0 abc def ghi \nHost 515 0 24 \n"
            stderr = ""

        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    p = LlamaCppProvider(
        config={"llama_cpp_dir": "/path", "gguf_path": "m.gguf"}
    )
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions(ctx_length=4096))
    # A non-numeric fit-params column must fail loudly as a RuntimeError, not
    # escape as a raw ValueError.
    with pytest.raises(RuntimeError):
        m.memory()


# ---- llama_cpp descriptor / OAI listing ----


def test_llama_cpp_get_models_descriptors():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(config={"alias": "qwen3"})
    descs = p.getModelsDescriptors()
    assert [d.modelId for d in descs] == ["qwen3"]
    assert all(d.provider is p for d in descs)


def test_llama_cpp_get_oai_models_provider_ctx():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(config={"alias": "qwen3", "ctx_length": 8192})
    data = p.getOAIModels()
    assert len(data) == 1
    assert data[0]["id"] == "qwen3"
    assert data[0]["context_length"] == 8192
    assert data[0]["object"] == "model"


def test_llama_cpp_get_oai_models_resident_ctx():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(config={"alias": "qwen3"})
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=16384))
    p.resident_model = m
    data = p.getOAIModels()
    assert data[0]["id"] == "qwen3"
    assert data[0]["context_length"] == 16384


def test_llama_cpp_get_oai_models_raises_without_ctx():
    from providers.llama_cpp import LlamaCppProvider

    # No provider ctx_length and no resident model: the context length is
    # genuinely unknown, so listing must fail loudly (ValueError) rather than
    # invent a default.
    p = LlamaCppProvider(config={"alias": "qwen3"})
    with pytest.raises(ValueError):
        p.getOAIModels()


# ---- llama_cpp load / unload ----


def test_llama_cpp_build_command():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/path/to/llama-cpp",
            "gguf_path": "model.gguf",
            "host": "0.0.0.0",
            "port": 9000,
            "ctx_length": 8192,
            "alias": "qwen3",
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    assert p._build_command(m) == [
        "/path/to/llama-cpp/llama-server",
        "-m",
        "model.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "-a",
        "qwen3",
        "-c",
        "8192",
    ]


def test_llama_cpp_build_command_omits_defaults():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/path",
            "gguf_path": "m.gguf",
            "alias": "a",
            "ctx_length": 4096,
        }
    )
    m = p.createModel(ModelDescriptor("a", p), LoadOptions(ctx_length=4096))
    # Default host/port are omitted so llama-server applies its own defaults.
    assert p._build_command(m) == [
        "/path/llama-server",
        "-m",
        "m.gguf",
        "-a",
        "a",
        "-c",
        "4096",
    ]


def test_llama_cpp_build_command_preserves_spaces():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "model with space.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    assert p._build_command(m) == [
        "/tmp/llama/llama-server",
        "-m",
        "model with space.gguf",
        "-a",
        "qwen3",
        "-c",
        "4096",
    ]


def test_llama_cpp_build_command_extra_options():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
            "extra_options": "--lora ./lora.bin -ngl 24 --cache-type-k q8_0 --port 9091",
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    command = p._build_command(m)
    # extra_options is inserted verbatim (tokenized) last, after YAALLB's own
    # -c, so it can override every YAALLB default (including -c and flags like
    # --port even when YAALLB emitted a default one).
    assert command == [
        "/tmp/llama/llama-server",
        "-m",
        "m.gguf",
        "-a",
        "qwen3",
        "-c",
        "4096",
        "--lora",
        "./lora.bin",
        "-ngl",
        "24",
        "--cache-type-k",
        "q8_0",
        "--port",
        "9091",
    ]


def test_llama_cpp_build_command_extra_options_with_spaces(monkeypatch):
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
            "extra_options": '--lora "lora with space.bin" --tags "tag a,b"',
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    command = p._build_command(m)
    # Quoted values with spaces survive tokenization.
    assert "--lora" in command
    assert "lora with space.bin" in command
    assert "--tags" in command
    assert "tag a,b" in command


def test_llama_cpp_build_command_core_options():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
            "options": {"ngl": 24, "cache_type_k": "q8_0", "lora": "lora.bin"},
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    assert p._build_command(m) == [
        "/tmp/llama/llama-server",
        "-m",
        "m.gguf",
        "-a",
        "qwen3",
        "--gpu-layers",
        "24",
        "--cache-type-k",
        "q8_0",
        "--lora",
        "lora.bin",
        "-c",
        "4096",
    ]


def test_llama_cpp_build_command_core_options_booleans():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
            "options": {"swa_full": True, "cpu_moe": True, "mlock": True, "no_host": False},
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    # Flags emitted only when true; false flags are omitted.
    assert p._build_command(m) == [
        "/tmp/llama/llama-server",
        "-m",
        "m.gguf",
        "-a",
        "qwen3",
        "--swa-full",
        "--cpu-moe",
        "--mlock",
        "-c",
        "4096",
    ]


def test_llama_cpp_load_spawns_process(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    spawned = {}

    class FakeProcess:
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=0):
            pass

        def kill(self):
            pass

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return FakeProcess()

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

        return Resp()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("providers.llama_cpp.httpx.get", fake_get)

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
            "ctx_length": 4096,
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    p.loadModel(m)

    assert spawned["kwargs"]["cwd"] == "/tmp/llama"
    assert spawned["argv"] == [
        "/tmp/llama/llama-server",
        "-m",
        "m.gguf",
        "-a",
        "qwen3",
        "-c",
        "4096",
    ]
    assert p.resident_model is m
    assert p._process is not None

    p.unloadModel(m)
    assert p._process is None
    assert p.resident_model is None
    assert not m.loaded


def test_llama_cpp_unload_kills_on_terminate_timeout(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    class FakeProcess:
        def __init__(self):
            self.killed = False

        def terminate(self):
            pass

        def wait(self, timeout=None):
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("llama-server", timeout)
            return 0

        def poll(self):
            return 0 if self.killed else None

        def kill(self):
            self.killed = True

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

        return Resp()

    proc = FakeProcess()
    monkeypatch.setattr(
        "providers.llama_cpp.subprocess.Popen", lambda *a, **kw: proc
    )
    monkeypatch.setattr("providers.llama_cpp.httpx.get", fake_get)

    p = LlamaCppProvider(
        config={
            "llama_cpp_dir": "/tmp/llama",
            "gguf_path": "m.gguf",
            "alias": "qwen3",
        }
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=8192))
    p.loadModel(m)

    p.unloadModel(m)
    assert proc.killed
    assert p._process is None
    assert p.resident_model is None
    assert not m.loaded


def test_llama_cpp_load_requires_dir_and_gguf():
    from providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider()
    m = p.createModel(ModelDescriptor("alias", p), LoadOptions())
    with pytest.raises(ValueError):
        p.loadModel(m)


def test_llama_cpp_load_waits_for_server_ready(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    class FakeProcess:
        def terminate(self):
            pass

        def wait(self, timeout=0):
            pass

        def kill(self):
            pass

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeProcess())

    calls = []
    statuses = iter([503, 503, 200])

    def fake_get(url, headers=None):
        calls.append(url)
        class Resp:
            status_code = next(statuses)

        return Resp()

    monkeypatch.setattr("providers.llama_cpp.httpx.get", fake_get)
    monkeypatch.setattr("providers.llama_cpp.time.sleep", lambda *a: None)

    p = LlamaCppProvider(
        config={"llama_cpp_dir": "/tmp/llama", "gguf_path": "m.gguf", "alias": "qwen3"}
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    p.loadModel(m)

    # loadModel blocks until the spawned server actually accepts requests;
    # only then is the model marked loaded (ready), not immediately on spawn.
    assert m.loaded
    assert m.load_state == "ready"
    assert all(c.endswith("/v1/models") for c in calls)
    assert len(calls) == 3


def test_llama_cpp_load_ready_timeout_raises(monkeypatch):
    import subprocess
    from providers.llama_cpp import LlamaCppProvider

    class FakeProcess:
        def terminate(self):
            pass

        def wait(self, timeout=0):
            pass

        def kill(self):
            pass

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakeProcess())

    def fake_get(url, headers=None):
        class Resp:
            status_code = 503

        return Resp()

    monkeypatch.setattr("providers.llama_cpp.httpx.get", fake_get)
    monkeypatch.setattr("providers.llama_cpp.time.sleep", lambda *a: None)
    monkeypatch.setattr("providers.llama_cpp.LLAMA_CPP_READY_TIMEOUT", 0.01)

    p = LlamaCppProvider(
        config={"llama_cpp_dir": "/tmp/llama", "gguf_path": "m.gguf", "alias": "qwen3"}
    )
    m = p.createModel(ModelDescriptor("qwen3", p), LoadOptions(ctx_length=4096))
    with pytest.raises(RuntimeError):
        p.loadModel(m)

    assert not m.loaded
    assert m.load_state == "loading"
