import pytest
import sys
import os
from fastapi.testclient import TestClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.serving.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["memory_rss_mb"] < 300.0  # Strict RAM budget verification!

def test_single_predict_positive(client):
    payload = {
        "texts": ["This app update is genuinely fire and goated 🔥"],
        "enable_fallback": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 1
    item = data["results"][0]
    assert item["label"] == "positive"
    assert item["confidence"] >= 0.60
    assert item["fallback_required"] is False

def test_single_predict_negative(client):
    payload = {
        "texts": ["Customer service was totally mid and completely cooked"],
        "enable_fallback": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    item = data["results"][0]
    assert item["label"] == "negative"
    assert item["confidence"] >= 0.60

def test_batch_predict(client):
    posts = [
        "Bro this is a massive W 🙌",
        "Absolute garbage update, crashes every 5 seconds 🗑️",
        "The system update is scheduled for tomorrow at 3pm",
        "Super clean design and fast performance ✨",
        "Total scam, avoid this product at all costs 👎"
    ]
    payload = {
        "texts": posts,
        "enable_fallback": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5
    assert len(data["results"]) == 5
    # Batch latency should be under 50ms for 5 posts
    assert data["batch_latency_ms"] < 50.0

def test_sarcasm_marker_triggers_fallback(client):
    payload = {
        "texts": ["Oh yeah, totally amazing job crashing the server /s"],
        "enable_fallback": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    item = data["results"][0]
    assert item["fallback_required"] is True
    assert item["reason"] == "sarcasm_detected"

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] > 0
    assert data["total_posts_analyzed"] > 0

def test_alert_flags_and_sub_aspects(client):
    payload = {
        "texts": [
            "The app crashes constantly and lags terribly on mobile.",
            "I love the design but hate the price tag.",
            "Normal meeting notes for team sync."
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    
    # Item 1: Performance issue & risk complaint
    assert results[0]["is_performance_issue"] is True
    assert results[0]["is_risk_complaint"] is True
    assert results[0]["is_mixed"] is False

    # Item 2: Mixed sentiment
    assert results[1]["is_mixed"] is True
    assert results[1]["is_performance_issue"] is False

    # Item 3: Neutral notes
    assert results[2]["is_performance_issue"] is False
    assert results[2]["is_risk_complaint"] is False
    assert results[2]["is_mixed"] is False


