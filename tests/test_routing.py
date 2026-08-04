import pytest
import httpx
from fastapi.testclient import TestClient

import main
from abstractions.descriptor import ModelDescriptor
from abstractions.load_options import LoadOptions
from abstractions.model import Model as BaseModel
from abstractions.provider import Provider


class FakeProvider(Provider):
    class Model(BaseModel):
        def memory(self) -> float:
            return 0.0

    def __init__(self, endpoint_uri: str, model_ids: list[str]) -> None:
        self._endpoint_uri = endpoint_uri
        self._descriptors = [ModelDescriptor(m, self) for m in model_ids]

    @property
    def endpoint_uri(self) -> str:
        return self._endpoint_uri

    def getModelsDescriptors(self) -> list[ModelDescriptor]:
        return list(self._descriptors)

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


def test_list_models_concatenates(monkeypatch):
    prov_a = FakeProvider("http://a.example/v1", ["model-a"])
    prov_b = FakeProvider("http://b.example/v1", ["model-b"])
    main.PROVIDERS = [prov_a, prov_b]

    def fake_get(url: str):
        class Resp:
            def __init__(self, data):
                self._data = data

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        if url == "http://a.example/v1/models":
            return Resp({"object": "list", "data": [
                {"id": "model-a", "created": 1, "object": "model", "owned_by": "a"},
            ]})
        if url == "http://b.example/v1/models":
            return Resp({"object": "list", "data": [
                {"id": "model-b", "created": 2, "object": "model", "owned_by": "b"},
            ]})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("httpx.get", fake_get)

    resp = TestClient(main.app).get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [m["id"] for m in data] == ["model-a", "model-b"]
    assert all(m["object"] == "model" for m in data)
