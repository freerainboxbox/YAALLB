import json

import pytest
import httpx
from fastapi.testclient import TestClient

import main
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider
from scheduling import Scheduler


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
    main.SCHEDULER = None
    yield
    main.PROVIDERS = []
    main.SCHEDULER = None


class FakeResponse:
    def __init__(
        self,
        content=b'{"choices":[{"index":0}],"usage":{}}',
        status_code=200,
        content_type="application/json",
    ):
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": content_type}


class FakeStreamResponse:
    def __init__(self, chunks=(b'data: {"x":1}\n\n', b"data: [DONE]\n\n")):
        self.chunks = chunks
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}

    async def aiter_raw(self):
        for c in self.chunks:
            yield c

    async def aread(self):
        return b'{"error":{"message":"boom"}}'


class FakeAsyncClient:
    def __init__(self, nonstream=None, stream=None):
        self.nonstream = nonstream
        self.stream = stream
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        pass

    async def post(self, url, json, headers):
        self.calls.append(("post", url, json, headers))
        return self.nonstream

    def build_request(self, method, url, json, headers):
        return {"url": url, "json": json, "headers": headers}

    async def send(self, req, stream=False):
        self.calls.append(("send", req["url"], req["json"], req["headers"]))
        return self.stream


def test_chat_completions_forwards_to_provider(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(nonstream=FakeResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "max_tokens": 512},
        )

    assert resp.status_code == 200
    assert resp.json() == {"choices": [{"index": 0}], "usage": {}}
    method, url, json, headers = fake.calls[0]
    assert method == "post"
    assert url == "http://a.example/v1/chat/completions"
    assert json["model"] == "model-a"
    assert json["max_tokens"] == 512
    assert headers == {}

    # Model released once the upstream request completes.
    model = main.SCHEDULER.resident[0]
    assert main.SCHEDULER.in_flight[model] == 0


def test_chat_completions_sends_api_key_header(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.api_key = "sk-test"
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(nonstream=FakeResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post("/v1/chat/completions", json={"model": "model-a", "messages": []})

    method, url, json, headers = fake.calls[0]
    assert headers == {"Authorization": "Bearer sk-test"}


def test_chat_completions_forwards_upstream_status(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(
        nonstream=FakeResponse(b'{"error":{"message":"boom"}}', status_code=503)
    )
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": []}
        )

    assert resp.status_code == 503
    assert resp.json() == {"error": {"message": "boom"}}


def test_chat_completions_streams_sse(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "stream": True, "messages": []},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == b'data: {"x":1}\n\ndata: [DONE]\n\n'
    assert fake.calls[0][0] == "send"
    assert fake.calls[0][1] == "http://a.example/v1/chat/completions"


def test_graceful_shutdown_unloads_resident_models(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a._unloaded = []

    def recording_unload(model):
        prov_a._unloaded.append(model.descriptor.modelId)
        model._loaded = False

    prov_a.unloadModel = recording_unload
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(nonstream=FakeResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post("/v1/chat/completions", json={"model": "model-a", "messages": []})
        assert main.SCHEDULER.resident  # model loaded during request

    # Exiting the TestClient runs lifespan shutdown: flush + unload residents.
    assert prov_a._unloaded == ["model-a"]
    assert not main.SCHEDULER.resident[0].loaded


def test_chat_completions_not_found():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
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

    def fake_get(url: str, headers: dict | None = None):
        captured["url"] = url
        captured["headers"] = headers
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
    assert captured["headers"] == {}


def test_getoaimodels_sends_api_key_header(monkeypatch):
    from providers.lmstudio import LMStudioProvider

    provider = LMStudioProvider(config={"api_key": "sk-test"})
    captured = {}

    def fake_get(url: str, headers: dict | None = None):
        captured["headers"] = headers
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"object": "list", "data": []}

        return Resp()

    monkeypatch.setattr("httpx.get", fake_get)

    provider.getOAIModels()
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}


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


def test_set_iogpu_wired_limit_success(monkeypatch):
    calls = []
    current = 107520

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        if argv == ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]:
            class Read:
                returncode = 0
                stdout = str(current)
                stderr = ""

            return Read()
        class Write:
            returncode = 0
            stdout = ""
            stderr = ""

        return Write()

    monkeypatch.setattr(
        "main.shutil.which", lambda name: "/usr/sbin/sysctl" if name == "sysctl" else "/usr/sbin/sudo"
    )
    monkeypatch.setattr("main.subprocess.run", fake_run)

    assert main.set_iogpu_wired_limit(112640) is True
    write_argv, write_kw = calls[1]
    assert write_argv == [
        "/usr/sbin/sudo",
        "/usr/sbin/sysctl",
        "-w",
        "iogpu.wired_limit_mb=112640",
    ]
    assert write_kw["timeout"] == 10


def test_set_iogpu_wired_limit_already_set(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        class Read:
            returncode = 0
            stdout = "112640"
            stderr = ""

        return Read()

    monkeypatch.setattr(
        "main.shutil.which", lambda name: "/usr/sbin/sysctl" if name == "sysctl" else "/usr/sbin/sudo"
    )
    monkeypatch.setattr("main.subprocess.run", fake_run)

    assert main.set_iogpu_wired_limit(112640) is True
    # No write attempted: only the read happened.
    assert calls == [["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]]


def test_set_iogpu_wired_limit_not_permitted(monkeypatch):
    def fake_run(argv, **kw):
        if argv == ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]:
            class Read:
                returncode = 0
                stdout = "107520"
                stderr = ""

            return Read()
        class Write:
            returncode = 1
            stdout = ""
            stderr = "Operation not permitted"

        return Write()

    monkeypatch.setattr(
        "main.shutil.which", lambda name: "/usr/sbin/sysctl" if name == "sysctl" else "/usr/sbin/sudo"
    )
    monkeypatch.setattr("main.subprocess.run", fake_run)

    assert main.set_iogpu_wired_limit(112640) is False


def test_set_iogpu_wired_limit_unreadable_still_writes(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv == ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"]:
            class Read:
                returncode = 1
                stdout = ""
                stderr = ""

            return Read()
        class Write:
            returncode = 0
            stdout = ""
            stderr = ""

        return Write()

    monkeypatch.setattr(
        "main.shutil.which", lambda name: "/usr/sbin/sysctl" if name == "sysctl" else "/usr/sbin/sudo"
    )
    monkeypatch.setattr("main.subprocess.run", fake_run)

    assert main.set_iogpu_wired_limit(112640) is True
    assert calls == [
        ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
        [
            "/usr/sbin/sudo",
            "/usr/sbin/sysctl",
            "-w",
            "iogpu.wired_limit_mb=112640",
        ],
    ]


def test_set_iogpu_wired_limit_no_sysctl(monkeypatch):
    monkeypatch.setattr("main.shutil.which", lambda _: None)
    assert main.set_iogpu_wired_limit(112640) is False


def test_set_iogpu_wired_limit_no_sudo(monkeypatch):
    def fake_run(argv, **kw):
        class Read:
            returncode = 0
            stdout = "107520"
            stderr = ""

        return Read()

    monkeypatch.setattr(
        "main.shutil.which", lambda name: "/usr/sbin/sysctl" if name == "sysctl" else None
    )
    monkeypatch.setattr("main.subprocess.run", fake_run)

    assert main.set_iogpu_wired_limit(112640) is False
