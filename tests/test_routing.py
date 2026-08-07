import json

import pytest
import httpx
from fastapi.testclient import TestClient

import main
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


class FakeProvider(Provider):
    _type_id = "fake"

    class Model(BaseModel):
        def memory(self) -> float:
            return 0.0

    def __init__(self, endpoint_uri: str, model_ids: list[str]) -> None:
        self._endpoint_uri = endpoint_uri
        self._model_ids = list(model_ids)
        self._descriptors = [ModelDescriptor(m, self) for m in model_ids]

    @property
    def endpoint_uri(self) -> str:
        return self._endpoint_uri

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        return list(self._descriptors)

    def getOAIModels(self) -> list[dict]:
        return [
            {"id": m, "object": "model", "created": 1, "owned_by": "fake"}
            for m in self._model_ids
        ]

    def createModel(self, descriptor, loadOptions) -> BaseModel:
        return self.Model(descriptor, loadOptions)

    def loadModel(self, model: BaseModel) -> None:
        model._loaded = True

    def unloadModel(self, model: BaseModel) -> None:
        model._loaded = False


@pytest.fixture(autouse=True)
def reset_providers():
    main.PROVIDERS = []
    yield
    main.PROVIDERS = []


def test_chat_completions_happy_path():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_b = FakeProvider("http://b.example/v1", ["model-b"])
    main.PROVIDERS = [prov_a, prov_b]

    resp = TestClient(main.app).post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": []},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "model-a"
    assert body["provider"] == prov_a.endpoint_uri
    assert body["status"] == "routed"


def test_chat_completions_not_found():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]

    resp = TestClient(main.app).post(
        "/v1/chat/completions",
        json={"model": "does-not-exist"},
    )

    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["type"] == "invalid_request_error"
    assert body["param"] == "model"
    assert body["code"] == "model_not_found"
    assert "does-not-exist" in body["message"]


def test_list_models_concatenates():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_b = FakeProvider("http://b.example/v1", ["model-b"])
    main.PROVIDERS = [prov_a, prov_b]

    resp = TestClient(main.app).get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [m["id"] for m in data] == ["model-a", "model-b"]
    assert all(m["object"] == "model" for m in data)


def test_getoaimodels_default_queries_endpoint(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    provider = LMStudioProvider()
    captured = {}

    def fake_get(url: str):
        captured["url"] = url
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"object": "list", "data": [
                    {"id": "x", "object": "model", "created": 1, "owned_by": "o"},
                ]}

        return Resp()

    monkeypatch.setattr("httpx.get", fake_get)

    data = provider.getOAIModels()
    assert data == [{"id": "x", "object": "model", "created": 1, "owned_by": "o"}]
    assert captured["url"] == "http://127.0.0.1:1234/v1/models"


def test_dwarfstar_getoaimodels_hardcoded(monkeypatch):
    from providers.dwarfstar import DwarfStarProvider

    def no_network(url):
        raise AssertionError("should not hit the network")

    monkeypatch.setattr("httpx.get", no_network)

    provider = DwarfStarProvider()
    data = provider.getOAIModels()
    assert [m["id"] for m in data] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert all(m["object"] == "model" for m in data)


def test_dwarfstar_resident_model_and_context(monkeypatch):
    from providers.dwarfstar import DwarfStarProvider

    def no_network(url):
        raise AssertionError("should not hit the network")

    monkeypatch.setattr("httpx.get", no_network)

    class FakeProcess:
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=0):
            pass

        def poll(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(
        "providers.dwarfstar.subprocess.Popen", lambda *a, **kw: FakeProcess()
    )

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "model.gguf"}
    )
    assert [m["context_length"] for m in provider.getOAIModels()] == [1000000, 1000000]

    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )
    provider.loadModel(model)
    assert provider.resident_model is model
    assert [m["context_length"] for m in provider.getOAIModels()] == [8192, 8192]

    provider.unloadModel(model)
    assert getattr(provider, "resident_model", None) is None
    assert [m["context_length"] for m in provider.getOAIModels()] == [1000000, 1000000]


def test_dwarfstar_build_command():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={
            "ds4_dir": "/path/to/ds4",
            "gguf_path": "./ds4flash-0731.gguf",
            "options": {"kv_disk_dir": "/tmp/ds4-0731-kv", "kv_disk_space_mb": 262144},
        }
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=1000000)
    )

    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "./ds4flash-0731.gguf",
        "--kv-disk-dir",
        "/tmp/ds4-0731-kv",
        "--kv-disk-space-mb",
        "262144",
        "--ctx",
        "1000000",
    ]


def test_dwarfstar_build_command_omits_defaults():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf"})
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=4096)
    )

    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "m.gguf",
        "--ctx",
        "4096",
    ]


def test_dwarfstar_build_command_overrides():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={
            "ds4_dir": "/tmp/ds4",
            "gguf_path": "m.gguf",
            "host": "0.0.0.0",
            "port": 9000,
            "options": {"power": 50, "cors": True, "tokens": 2048},
        }
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=4096)
    )

    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "m.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "-n",
        "2048",
        "--power",
        "50",
        "--cors",
        "--ctx",
        "4096",
    ]


def test_dwarfstar_load_spawns_process(monkeypatch):
    import subprocess
    from providers.dwarfstar import DwarfStarProvider

    spawned = {}

    class FakeProcess:
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=0):
            pass

        def poll(self):
            return 0

        def kill(self):
            pass

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf"}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=4096)
    )
    provider.loadModel(model)

    assert spawned["kwargs"]["cwd"] == "/tmp/ds4"
    assert spawned["argv"] == [
        "./ds4-server",
        "-m",
        "m.gguf",
        "--ctx",
        "4096",
    ]
    assert provider.resident_model is model
    assert provider._process is not None

    provider.unloadModel(model)
    assert provider._process is None
    assert provider.resident_model is None


def test_dwarfstar_load_requires_ds4_dir_and_gguf_path():
    import pytest
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider()
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions()
    )
    with pytest.raises(ValueError):
        provider.loadModel(model)


def test_provider_type_ids():
    from providers.dwarfstar import DwarfStarProvider
    from providers.lmstudio import LMStudioProvider

    assert LMStudioProvider()._type_id == "lms"
    assert DwarfStarProvider()._type_id == "ds4"
    assert LMStudioProvider()._instance_id == 0


def test_load_providers_from_config(tmp_path):
    config = {
        "lms": [{"host": "10.0.0.1", "port": 9999}],
        "ds4": [{}, {"host": "0.0.0.0", "port": 8080}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    providers = main.load_providers(str(path))

    assert [p._type_id for p in providers] == ["lms", "ds4", "ds4"]
    assert providers[0]._instance_id == 0
    assert providers[0].host == "10.0.0.1"
    assert providers[0].port == 9999
    assert providers[1]._instance_id == 0
    assert providers[1].host == "127.0.0.1"
    assert providers[1].port == 8000
    assert providers[2]._instance_id == 1
    assert providers[2].host == "0.0.0.0"
    assert providers[2].port == 8080


def test_load_providers_disabled_types(tmp_path):
    config = {"lms": [{}], "unknown": [{}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    providers = main.load_providers(str(path))

    assert len(providers) == 1
    assert providers[0]._type_id == "lms"


def test_load_providers_missing_config(tmp_path):
    providers = main.load_providers(str(tmp_path / "nope.json"))
    assert providers == []
