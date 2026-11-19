import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.serving.app import app  # noqa: E402

BODY = {"texts": ["Loved the new update!"], "extract_aspects": False}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_no_keys_configured_means_open(client, monkeypatch):
    monkeypatch.delenv("SENTIMENT_API_KEYS", raising=False)
    assert client.post("/predict", json=BODY).status_code == 200


def test_missing_and_wrong_key_rejected(client, monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one,key-two")
    r = client.post("/predict", json=BODY)
    assert r.status_code == 401 and r.headers.get("www-authenticate") == "Bearer"
    assert client.post("/predict", json=BODY, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/predict", json=BODY, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_valid_keys_accepted_both_header_styles(client, monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one, key-two")
    assert client.post("/predict", json=BODY, headers={"X-API-Key": "key-one"}).status_code == 200
    assert client.post("/predict", json=BODY, headers={"Authorization": "Bearer key-two"}).status_code == 200
    assert client.post("/api/v1/sentiment/predict", json=BODY, headers={"X-API-Key": "key-two"}).status_code == 200


def test_other_data_endpoints_protected(client, monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one")
    assert client.get("/metrics").status_code == 401
    assert client.get("/api/v1/sentiment/active_learning").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "key-one"}).status_code == 200


def test_health_and_dashboard_stay_public(client, monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one")
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("route", ["/analyze/file", "/analyze/image", "/analyze/video", "/analyze/document"])
def test_analyze_endpoints_protected(client, monkeypatch, route):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one")
    r = client.post(route, files={"file": ("note.txt", b"great product", "text/plain")})
    assert r.status_code == 401


def test_valid_bearer_wins_over_wrong_x_api_key(client, monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEYS", "key-one")
    headers = {"X-API-Key": "wrong", "Authorization": "Bearer key-one"}
    assert client.post("/predict", json=BODY, headers=headers).status_code == 200
