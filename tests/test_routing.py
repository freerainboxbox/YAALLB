import json

import pytest
import httpx
from fastapi.testclient import TestClient

import main
from main import DEFAULT_CTX_LENGTH, STARTUP_ATTEMPTS
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


def test_chat_completions_max_tokens_does_not_drive_ctx(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "max_tokens": 512, "stream": True},
        )

    assert resp.status_code == 200
    # max_tokens bounds the number of tokens emitted, not the model's context
    # window; it must not drive context sizing.
    assert main.SCHEDULER.resident[0].loadOptions.ctx_length == DEFAULT_CTX_LENGTH
    method, url, json, headers = fake.calls[0]
    assert json["max_tokens"] == 512


def test_chat_completions_descriptor_lookup_runs_off_thread(monkeypatch):
    import asyncio as aio

    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    on_loop = []
    orig = prov_a.getModelsDescriptors

    def record():
        try:
            aio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return orig()

    prov_a.getModelsDescriptors = record

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    assert resp.status_code == 200
    # A provider's descriptor lookup can block on HTTP (LM Studio TTL miss);
    # it must never run on the event-loop thread, from either the route's
    # lookup_model or the scheduler's _serve path.
    assert on_loop
    assert not any(on_loop)


def test_chat_completions_forwards_to_provider(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "max_tokens": 512, "stream": True},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content.startswith(b'data: {"status": "processing"')
    assert resp.content.endswith(b'data: {"x":1}\n\ndata: [DONE]\n\n')
    method, url, json, headers = fake.calls[0]
    assert method == "send"
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

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    method, url, json, headers = fake.calls[0]
    assert headers == {"Authorization": "Bearer sk-test"}


def test_chat_completions_upstream_not_ready_sse_error(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    class NonReadyStream:
        status_code = 503
        headers = {"content-type": "text/event-stream"}

    fake = FakeAsyncClient(stream=NonReadyStream())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("main.asyncio.sleep", no_sleep)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )

    # A non-200 upstream during startup becomes an SSE error event, not a 503.
    assert resp.status_code == 200
    assert b'"code": "provider_start_failed"' in resp.content
    assert prov_a.startup_failures == STARTUP_ATTEMPTS


def test_chat_completions_connection_error_retries_then_sse_error(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    class RaisingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aclose(self):
            pass

        def build_request(self, method, url, json, headers):
            return {"url": url, "json": json, "headers": headers}

        async def send(self, req, stream=False):
            raise httpx.ConnectError("ds4-server not up")

    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: RaisingClient())

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("main.asyncio.sleep", no_sleep)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )

    # Connection failures retry internally, then surface as an SSE error event.
    assert resp.status_code == 200
    assert b'"code": "provider_start_failed"' in resp.content
    assert prov_a.startup_failures == STARTUP_ATTEMPTS


def test_chat_completions_success_resets_startup_failures(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    prov_a.startup_failures = 9
    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )

    assert prov_a.startup_failures == 0


def test_chat_completions_lmstudio_404_waits_ready_then_succeeds(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a._type_id = "lms"
    waited = []
    prov_a.wait_loaded = lambda model_id: waited.append(model_id)
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    class FlakyClient:
        def __init__(self):
            self.calls = []
            self.responses = [404, 200]

        async def aclose(self):
            pass

        def build_request(self, method, url, json, headers):
            return {"url": url, "json": json, "headers": headers}

        async def send(self, req, stream=False):
            self.calls.append(("send", req["url"], req["json"], req["headers"]))
            status = self.responses.pop(0)
            if status == 404:
                class R:
                    status_code = 404
                    headers = {"content-type": "text/event-stream"}

                return R()
            return FakeStreamResponse()

    fake = FlakyClient()
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )

    # An LM Studio "still loading" 404 must NOT be treated as a startup
    # failure: YAALLB blocks on the model's real readiness signal, then
    # retries and succeeds. startup_failures is never bumped.
    assert resp.status_code == 200
    assert resp.content.endswith(b'data: {"x":1}\n\ndata: [DONE]\n\n')
    assert waited == ["model-a"]
    assert len(fake.calls) == 2
    assert getattr(prov_a, "startup_failures", 0) == 0


def test_chat_completions_lmstudio_404_never_ready_sse_error(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a._type_id = "lms"

    def failing_wait(model_id):
        raise RuntimeError("never ready")

    prov_a.wait_loaded = failing_wait
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    class Always404:
        status_code = 404
        headers = {"content-type": "text/event-stream"}

    fake = FakeAsyncClient(stream=Always404())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )

    # A 404 that never becomes ready surfaces a distinct SSE error, NOT the
    # generic provider_start_failed (which conflates it with provider startup).
    assert resp.status_code == 200
    assert b'"code": "model_not_ready"' in resp.content
    assert getattr(prov_a, "startup_failures", 0) == 0


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
    # Prelim event comes first, then the relayed upstream chunks.
    assert resp.content == b'data: {"status": "processing", "model": "model-a", "choices": []}\n\ndata: {"x":1}\n\ndata: [DONE]\n\n'
    assert fake.calls[0][0] == "send"
    assert fake.calls[0][1] == "http://a.example/v1/chat/completions"


def test_chat_completions_rejects_stream_false():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": []},
        )

    assert resp.status_code == 400
    body = resp.json()["error"]
    assert body["code"] == "stream_required"
    assert body["param"] == "stream"
    assert "stream" in body["message"]


def test_chat_completions_non_streaming_direct_forward_passthrough(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"allow_non_streaming": True}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    resp404 = FakeResponse(
        content=b'{"error":"boom"}', status_code=404, content_type="application/json"
    )
    fake = FakeAsyncClient(nonstream=resp404, stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": False},
        )

    # Direct forward returns the upstream status/body as-is (no success
    # guarantee, no SSE, no retry loop).
    assert resp.status_code == 404
    assert resp.content == b'{"error":"boom"}'
    method, url, json, headers = fake.calls[0]
    assert method == "post"
    assert url == "http://a.example/v1/chat/completions"
    assert json["stream"] is False
    # Model is released once the direct forward completes.
    assert main.SCHEDULER.in_flight[main.SCHEDULER.resident[0]] == 0


def test_chat_completions_allow_non_streaming_keeps_streaming(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"allow_non_streaming": True}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    # allow_non_streaming does NOT downgrade an explicit stream=true request.
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert fake.calls[0][0] == "send"


def test_chat_completions_non_streaming_model_rejects_stream(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"supports_streaming": False}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    # A model that cannot stream must error out on a stream=true request.
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "model_does_not_support_streaming"


def test_chat_completions_non_streaming_model_requires_allow_non_streaming(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"supports_streaming": False}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": False},
        )

    # A non-streaming model needs allow_non_streaming=true to serve stream=false.
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "stream_required"


def test_chat_completions_non_streaming_model_with_allow_forwards(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {
        "model-a": {"supports_streaming": False, "allow_non_streaming": True}
    }
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    resp404 = FakeResponse(
        content=b'{"error":"boom"}', status_code=404, content_type="application/json"
    )
    fake = FakeAsyncClient(nonstream=resp404, stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": False},
        )

    # A non-streaming model with allow_non_streaming=true serves stream=false
    # via the direct-forward path (upstream status passed through as-is).
    assert resp.status_code == 404
    assert resp.content == b'{"error":"boom"}'
    assert fake.calls[0][0] == "post"


def test_graceful_shutdown_unloads_resident_models(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a._unloaded = []

    def recording_unload(model):
        prov_a._unloaded.append(model.descriptor.modelId)
        model._loaded = False

    prov_a.unloadModel = recording_unload
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post(
            "/v1/chat/completions", json={"model": "model-a", "messages": [], "stream": True}
        )
        assert main.SCHEDULER.resident  # model loaded during request

    # Exiting the TestClient runs lifespan shutdown: flush + unload residents.
    assert prov_a._unloaded == ["model-a"]
    assert not main.SCHEDULER.resident[0].loaded


def test_chat_completions_scheduler_failure_sse_error(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])

    def failing_load(model):
        raise RuntimeError("boom load")

    prov_a.loadModel = failing_load
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    # A scheduler-side load failure becomes an SSE error event, not an abrupt
    # stream abort after the prelim 200 was already committed.
    assert resp.status_code == 200
    assert b'"code": "model_load_failed"' in resp.content


def test_chat_completions_not_found():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "does-not-exist", "stream": True},
        )

    # Model-not-found surfaces as an SSE error event (still 200) so the
    # client doesn't reject the response outright.
    assert resp.status_code == 200
    body = resp.content
    assert b'"code": "model_not_found"' in body
    assert "does-not-exist" in resp.text


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

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

        return Resp()

    monkeypatch.setattr("httpx.get", fake_get)

    class FakeProcess:
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=0):
            pass

        def poll(self):
            return None

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


def test_dwarfstar_build_command_uses_load_options_ctx():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={"ds4_dir": "/path/to/ds4", "gguf_path": "./ds4flash-0731.gguf"}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )

    # With no provider-level ctx_length, the spawned --ctx comes from the
    # loading model's load options, matching what memory() and /v1/models
    # report once the model is resident.
    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "./ds4flash-0731.gguf",
        "--ctx",
        "8192",
    ]


def test_dwarfstar_provider_ctx_overrides_model_ctx():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={
            "ds4_dir": "/path/to/ds4",
            "gguf_path": "./ds4flash-0731.gguf",
            "ctx_length": 262144,
        }
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=1000000)
    )

    # Provider-level ctx_length wins over the model's load options.
    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "./ds4flash-0731.gguf",
        "--ctx",
        "262144",
    ]
    # And both served models report it in /v1/models.
    assert [m["context_length"] for m in provider.getOAIModels()] == [262144, 262144]


def test_dwarfstar_build_command_preserves_spaces():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={
            "ds4_dir": "/tmp/ds4",
            "gguf_path": "model with space.gguf",
            "ctx_length": 4096,
        }
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=4096)
    )

    # A gguf_path containing spaces must stay a single argv token, not be
    # re-split by shell word-splitting.
    assert provider._build_command(model) == [
        "./ds4-server",
        "-m",
        "model with space.gguf",
        "--ctx",
        "4096",
    ]


def test_dwarfstar_build_command_omits_defaults():
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf", "ctx_length": 4096}
    )
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
            "ctx_length": 4096,
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
            return None

        def kill(self):
            pass

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return FakeProcess()

    def fake_get(url, headers=None):
        class Resp:
            status_code = 200

        return Resp()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("providers.dwarfstar.httpx.get", fake_get)

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf", "ctx_length": 4096}
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


def test_dwarfstar_unload_kills_on_terminate_timeout(monkeypatch):
    import subprocess
    from providers.dwarfstar import DwarfStarProvider

    class FakeProcess:
        def __init__(self):
            self.killed = False

        def terminate(self):
            pass

        def wait(self, timeout=None):
            # A ds4 stuck past the terminate timeout raises TimeoutExpired;
            # after kill() the reaped wait returns normally.
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("ds4-server", timeout)
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
        "providers.dwarfstar.subprocess.Popen", lambda *a, **kw: proc
    )
    monkeypatch.setattr("providers.dwarfstar.httpx.get", fake_get)

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "model.gguf"}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )
    provider.loadModel(model)

    # Must escalate to kill() instead of propagating TimeoutExpired, so the
    # scheduler's eviction path still tears down state.
    provider.unloadModel(model)
    assert proc.killed
    assert provider._process is None
    assert provider.resident_model is None
    assert not model.loaded


def test_dwarfstar_load_requires_ds4_dir_and_gguf_path():
    import pytest
    from providers.dwarfstar import DwarfStarProvider

    provider = DwarfStarProvider()
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions()
    )
    with pytest.raises(ValueError):
        provider.loadModel(model)


def test_dwarfstar_load_waits_for_server_ready(monkeypatch):
    import subprocess
    from providers.dwarfstar import DwarfStarProvider

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

    monkeypatch.setattr("providers.dwarfstar.httpx.get", fake_get)
    monkeypatch.setattr("providers.dwarfstar.time.sleep", lambda *a: None)

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf"}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )
    provider.loadModel(model)

    # loadModel blocks until the spawned ds4-server actually accepts requests;
    # only then is the model marked loaded (ready), not immediately on spawn.
    assert model.loaded
    assert model.load_state == "ready"
    assert all(c.endswith("/v1/models") for c in calls)
    assert len(calls) == 3


def test_dwarfstar_load_ready_timeout_raises(monkeypatch):
    import subprocess
    from providers.dwarfstar import DwarfStarProvider

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

    monkeypatch.setattr("providers.dwarfstar.httpx.get", fake_get)
    monkeypatch.setattr("providers.dwarfstar.time.sleep", lambda *a: None)
    monkeypatch.setattr("providers.dwarfstar.DS4_READY_TIMEOUT", 0.01)

    provider = DwarfStarProvider(
        config={"ds4_dir": "/tmp/ds4", "gguf_path": "m.gguf"}
    )
    model = provider.createModel(
        ModelDescriptor("deepseek-v4-flash", provider), LoadOptions(ctx_length=8192)
    )
    with pytest.raises(RuntimeError):
        provider.loadModel(model)

    assert not model.loaded
    assert model.load_state == "loading"


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


def test_main_wires_scheduler_globals(tmp_path, monkeypatch):
    config = {"vram_limit_mb": 112640, "lms": [{"host": "127.0.0.1", "port": 1234}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    started = {}

    def fake_uvicorn_run(app, **kw):
        started["app"] = app
        started["host"] = kw["host"]
        started["port"] = kw["port"]

    monkeypatch.setattr("main.uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr(
        "main.set_iogpu_wired_limit", lambda _: True
    )
    monkeypatch.setattr(
        "sys.argv", ["yaallb", "--config", str(path)]
    )

    main.main()

    # The module globals must be assigned so routes can use them at runtime.
    assert main.SCHEDULER is not None
    assert main.PROVIDERS != []
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 4343


def test_load_yaallb_config_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"vram_limit_mb": 1}))
    cfg = main.load_yaallb_config(str(path))
    assert cfg == {"address": "127.0.0.1", "port": 4343, "ctx_length": 4096}


def test_load_yaallb_config_reads_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"yaallb": {"address": "0.0.0.0", "port": 8000, "ctx_length": 8192}}
        )
    )
    cfg = main.load_yaallb_config(str(path))
    assert cfg == {"address": "0.0.0.0", "port": 8000, "ctx_length": 8192}


def test_load_yaallb_config_missing_file(tmp_path):
    cfg = main.load_yaallb_config(str(tmp_path / "nope.json"))
    assert cfg == {"address": "127.0.0.1", "port": 4343, "ctx_length": 4096}


def test_model_overrides_for():
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {
        "model-a": {"ctx_length": 8192, "temperature": 0.7}
    }
    assert main.model_overrides_for(prov_a, "model-a") == {
        "ctx_length": 8192,
        "temperature": 0.7,
    }
    assert main.model_overrides_for(prov_a, "model-b") == {}
    assert main.model_overrides_for(FakeProvider("http://b/v1", ["x"]), "x") == {}


def test_chat_completions_applies_ctx_and_override(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"ctx_length": 8192, "temperature": 0.7}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        )

    assert resp.status_code == 200
    method, url, json, headers = fake.calls[0]
    # ctx_length came from the model override (request specified neither).
    assert main.SCHEDULER.resident[0].loadOptions.ctx_length == 8192
    # temperature was injected into the forwarded body.
    assert json["temperature"] == 0.7
    # ctx_length itself is not forwarded to the upstream.
    assert "ctx_length" not in json


def test_chat_completions_client_ctx_wins_over_override(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_a.model_overrides = {"model-a": {"ctx_length": 8192, "temperature": 0.7}}
    main.PROVIDERS = [prov_a]
    main.SCHEDULER = Scheduler(main.PROVIDERS, 24576)

    fake = FakeAsyncClient(stream=FakeStreamResponse())
    monkeypatch.setattr("main.httpx.AsyncClient", lambda *a, **kw: fake)

    with TestClient(main.app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "context_length": 4096, "messages": [], "stream": True},
        )

    assert main.SCHEDULER.resident[0].loadOptions.ctx_length == 4096
    method, url, json, headers = fake.calls[0]
    assert json["context_length"] == 4096
    assert json["temperature"] == 0.7
