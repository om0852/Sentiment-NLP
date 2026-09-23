import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi.testclient import TestClient
from src.serving.app import app
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.serving.cache import SentimentLRUCache

def run_tests():
    print("=" * 60)
    print("RUNNING VERIFICATION FOR OPTION 1 & OPTION 2")
    print("=" * 60)

    # 1. Unit Test ContextAnalyzer directly
    analyzer = ContextAnalyzer()
    print(f"Loaded {len(analyzer.cultural_disasters)} disaster tropes and {len(analyzer.cultural_triumphs)} triumph tropes.")
    assert len(analyzer.cultural_disasters) > 0, "No disaster tropes loaded!"
    assert len(analyzer.cultural_triumphs) > 0, "No triumph tropes loaded!"

    # Test 1.1: Cultural Disasters (Option 1)
    disaster_samples = [
        ("They really pulled a Season 8 Game of Thrones on this update", "negative", "cultural_disaster_metaphor"),
        ("This app makes me feel like a 19th-century chimney sweep on a Tuesday afternoon", "negative", "cultural_disaster_metaphor"),
        ("Today's deploy gave pure CrowdStrike update vibes", "negative", "cultural_disaster_metaphor"),
        ("This release was a complete Fyre Festival", "negative", "cultural_disaster_metaphor"),
        ("Experiencing the blue screen of death every 10 minutes", "negative", "cultural_disaster_metaphor")
    ]
    for text, expected_label, expected_reason in disaster_samples:
        lbl, conf, _, reason = analyzer.analyze(text, "neutral", 0.50)
        print(f"[TEST 1.1] Disasters: '{text[:40]}...' -> {lbl} ({reason})")
        assert lbl == expected_label, f"Failed for '{text}': got {lbl} != {expected_label}"
        assert reason == expected_reason, f"Wrong reason for '{text}': got {reason}"

    # Test 1.2: Cultural Triumphs (Option 1)
    triumph_samples = [
        ("The new UI redesign is literally the Mona Lisa", "positive", "cultural_triumph_metaphor"),
        ("This feature is an absolute masterpiece chef's kiss", "positive", "cultural_triumph_metaphor")
    ]
    for text, expected_label, expected_reason in triumph_samples:
        lbl, conf, _, reason = analyzer.analyze(text, "neutral", 0.50)
        print(f"[TEST 1.2] Triumphs: '{text[:40]}...' -> {lbl} ({reason})")
        assert lbl == expected_label, f"Failed for '{text}': got {lbl} != {expected_label}"
        assert reason == expected_reason, f"Wrong reason for '{text}': got {reason}"

    # Test 1.3: Negation of Disaster Trope (Litotes / Double Negation)
    negated_sample = "At least this update is not a dumpster fire like last month"
    lbl, conf, _, reason = analyzer.analyze(negated_sample, "positive", 0.70)
    print(f"[TEST 1.3] Negated Trope: '{negated_sample}' -> {lbl} ({reason})")
    assert reason != "cultural_disaster_metaphor", "Negated disaster trope incorrectly flagged as disaster!"

    # Test 2.1: Context Mismatch Sarcasm (Option 2)
    sarcasm_pairs = [
        ("Brilliant job guys, truly remarkable 👏", "Major database outage affecting all production servers", "negative", "context_mismatch_sarcasm"),
        ("Love to see it, 10/10 update 🔥", "Payment API down and checkout completely broken", "negative", "context_mismatch_sarcasm"),
        ("Congrats on this stellar achievement", "App crashing on launch for 80% of iOS users", "negative", "context_mismatch_sarcasm")
    ]
    for text, ctx, expected_label, expected_reason in sarcasm_pairs:
        lbl, conf, _, reason = analyzer.analyze(text, "positive", 0.95, context=ctx)
        print(f"[TEST 2.1] Context Sarcasm: '{text}' [Ctx: '{ctx}'] -> {lbl} ({reason})")
        assert lbl == expected_label, f"Failed for '{text}': got {lbl} != {expected_label}"
        assert reason == expected_reason, f"Wrong reason for '{text}': got {reason}"

    # Test 2.2: Context Reinforced Negative
    lbl, conf, _, reason = analyzer.analyze("still not working on my end", "negative", 0.80, context="Server outage")
    print(f"[TEST 2.2] Context Reinforced: 'still not working' -> {lbl} ({reason})")
    assert lbl == "negative"
    assert reason == "context_reinforced_negative"

    # Test 3: LRU Cache Context Separation
    cache = SentimentLRUCache(max_size=100)
    cache.put("Great work!", {"label": "positive", "confidence": 0.99}, context=None)
    cache.put("Great work!", {"label": "negative", "confidence": 0.92}, context="Server Outage")
    
    no_ctx = cache.get("Great work!", context=None)
    with_ctx = cache.get("Great work!", context="Server Outage")
    assert no_ctx["label"] == "positive", "Cache without context returned wrong label!"
    assert with_ctx["label"] == "negative", "Cache with context returned wrong label!"
    print("[TEST 3] Cache Context Separation: PASSED")

    # Test 4: End-to-End FastAPI TestClient
    with TestClient(app) as client:
        # 4.1 Test health
        res = client.get("/health")
        assert res.status_code == 200
        print(f"[TEST 4.1] Health: {res.json()['status']} (Memory RSS: {res.json()['memory_rss_mb']} MB)")

        # 4.2 Test simple legacy texts mode
        payload_simple = {
            "texts": [
                "They really pulled a Season 8 Game of Thrones on this update",
                "The new dashboard is absolute masterpiece"
            ]
        }
        res = client.post("/predict", json=payload_simple)
        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["label"] == "negative"
        assert data["results"][0]["reason"] == "cultural_disaster_metaphor"
        assert data["results"][1]["label"] == "positive"
        assert data["results"][1]["reason"] == "cultural_triumph_metaphor"
        print(f"[TEST 4.2] Legacy texts endpoint: PASSED (latency: {data['batch_latency_ms']} ms)")

        # 4.3 Test new structured contextual posts mode (Option 2)
        payload_contextual = {
            "posts": [
                {
                    "text": "Brilliant job guys, love to see it 👏",
                    "context": "Major database outage affecting all production servers"
                },
                {
                    "text": "Brilliant job guys, love to see it 👏",
                    "context": "Successfully shipped the new high-speed caching engine"
                }
            ]
        }
        res = client.post("/predict", json=payload_contextual)
        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 2
        # First post under outage -> sarcastic negative
        assert data["results"][0]["label"] == "negative"
        assert data["results"][0]["reason"] == "context_mismatch_sarcasm"
        # Second post under success -> genuine positive
        assert data["results"][1]["label"] == "positive"
        print(f"[TEST 4.3] Contextual posts endpoint: PASSED (latency: {data['batch_latency_ms']} ms)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
