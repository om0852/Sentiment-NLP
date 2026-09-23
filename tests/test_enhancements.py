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

def test_complex_real_world_posts():
    with TestClient(app) as client:
        # Case 1: Double negation phone praise
        post1 = "Not gonna lie, I thought this phone would be mid but ngl it's actually not bad at all, kinda impressed ngl"
        res1 = client.post("/predict", json={"texts": [post1]}).json()
        item1 = res1["results"][0]
        assert item1["label"] == "positive"
        assert item1["probabilities"]["positive"] > 0.80

        # Case 2: Flight delayed heavy sarcasm with 🙃 and 💀
        post2 = "Oh great, my flight got delayed by 6 hours AGAIN 🙃 love spending my birthday in an airport lounge eating stale sandwiches, living my best life fr fr 💀"
        res2 = client.post("/predict", json={"texts": [post2]}).json()
        item2 = res2["results"][0]
        assert item2["label"] == "negative"
        assert item2["probabilities"]["negative"] > 0.85
        assert item2["reason"] == "sarcastic_irony_detected"

def test_precision_and_advanced_sarcasm():
    with TestClient(app) as client:
        # Precision & Nuance: Objective News & Betting Tables
        post_news = "Cricket Betting Odds by SuperSports - SuperSportBet. Match Winner."
        res_news = client.post("/predict", json={"texts": [post_news]}).json()
        item_news = res_news["results"][0]
        assert item_news["label"] == "neutral"
        assert item_news["reason"] == "objective_news_or_listing"

        # Context Superiority: Conditional Trap Sarcasm
        post_trap = "Works great if your goal was to crash my entire browser every 5 minutes."
        res_trap = client.post("/predict", json={"texts": [post_trap]}).json()
        item_trap = res_trap["results"][0]
        assert item_trap["label"] == "negative"
        assert item_trap["reason"] == "conditional_trap_sarcasm"

        # Context Superiority: Faux Gratitude
        post_grat = "Thank you for reminding me why I cancelled my subscription last month."
        res_grat = client.post("/predict", json={"texts": [post_grat]}).json()
        item_grat = res_grat["results"][0]
        assert item_grat["label"] == "negative"
        assert item_grat["reason"] == "faux_gratitude_sarcasm"

        # Context Superiority: Passive-Aggressive Shoutouts
        post_shout = "Huge props to the dev team for breaking production right before the long weekend."
        res_shout = client.post("/predict", json={"texts": [post_shout]}).json()
        item_shout = res_shout["results"][0]
        assert item_shout["label"] == "negative"
        assert item_shout["reason"] == "passive_aggressive_praise"

        # Context Superiority: Rhetorical Imagine Derision
        post_imag = "Imagine charging $50/mo for a tool that can't even export a clean PDF."
        res_imag = client.post("/predict", json={"texts": [post_imag]}).json()
        item_imag = res_imag["results"][0]
        assert item_imag["label"] == "negative"
        assert item_imag["reason"] == "rhetorical_imagine_derision"


