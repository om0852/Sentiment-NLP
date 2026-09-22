import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from src.serving.app import app
from src.preprocessing.aspect_extractor import AspectExtractor
from src.serving.cache import SentimentLRUCache
from src.serving.active_learning import ActiveLearningQueue

def test_aspect_extractor():
    extractor = AspectExtractor()
    
    # Test UI positive + Performance negative
    text1 = "The new UI looks insanely clean and the animations slap, but half the buttons randomly stop working and the app crashes."
    aspects1 = extractor.extract_aspects(text1)
    assert "ui_ux" in aspects1
    assert aspects1["ui_ux"] == "positive"
    assert "performance" in aspects1
    assert aspects1["performance"] == "negative"
    
    # Test Pricing negative + Support negative + Features positive
    text2 = "Features are awesome and update is solid, but subscription pricing is a total ripoff and customer support ghosted me."
    aspects2 = extractor.extract_aspects(text2)
    assert aspects2.get("pricing_value") == "negative"
    assert aspects2.get("customer_support") == "negative"
    assert aspects2.get("features") == "positive"

def test_sentiment_lru_cache():
    cache = SentimentLRUCache(max_size=3)
    
    cache.put("Post A", {"label": "positive"})
    cache.put("Post B", {"label": "negative"})
    cache.put("Post C", {"label": "neutral"})
    
    # Check hit
    hit = cache.get("Post A")
    assert hit is not None
    assert hit["label"] == "positive"
    assert hit["cached"] is True
    
    # Eviction test: inserting Post D should evict Post B (since Post A was accessed)
    cache.put("Post D", {"label": "positive"})
    assert cache.get("Post B") is None
    assert cache.get("Post A") is not None
    assert cache.get("Post C") is not None
    assert cache.get("Post D") is not None
    
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1

def test_active_learning_queue():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        
    try:
        queue = ActiveLearningQueue(log_path=tmp_path)
        assert queue.count_queued() == 0
        
        queue.log_feedback(
            text="ambiguous review text",
            initial_label="neutral",
            confidence=0.51,
            reason="low_confidence"
        )
        assert queue.count_queued() == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_api_aspects_and_cache():
    with TestClient(app) as client:
        # First query (cold cache)
        req = {
            "texts": ["The new UI looks insanely clean and the animations slap, but half the buttons randomly stop working."],
            "extract_aspects": True
        }
        res1 = client.post("/predict", json=req)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["total_processed"] == 1
        item1 = data1["results"][0]
        assert "aspects" in item1
        assert item1["aspects"].get("ui_ux") == "positive"
        assert item1["cached"] is False
        
        # Second query (warm cache hit)
        res2 = client.post("/predict", json=req)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cache_hits"] == 1
        item2 = data2["results"][0]
        assert item2["cached"] is True
        assert item2["label"] == item1["label"]

def test_api_dashboard_and_metrics():
    with TestClient(app) as client:
        # Test dashboard HTML endpoint
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Sentiment Engine" in res.text
        
        # Test metrics endpoint includes cache_stats and active_learning_queued
        m_res = client.get("/metrics")
        assert m_res.status_code == 200
        m_data = m_res.json()
        assert "cache_stats" in m_data
        assert "active_learning_queued" in m_data
