import pytest
from starlette.testclient import TestClient
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sentiment-engine"))
sys.path.insert(0, project_root)
from src.serving.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_fancy_unicode_fonts(client):
    """Verifies that mathematical and stylized Unicode fonts are cleanly normalized."""
    payload = {
        "texts": [
            "This app is 𝕓𝕒𝕕 and utterly useless",
            "The new interface is 𝗕𝗘𝗦𝗧 and super smooth",
            "Customer support is 𝓫𝓪𝓭",
            "Speed is ˢᵘᵖᵉʳ fast"
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["label"] == "negative"
    assert results[1]["label"] == "positive"
    assert results[2]["label"] == "negative"
    assert results[3]["label"] == "positive"

def test_hashtag_splitting_sentiment(client):
    """Verifies that camelCase and underscored hashtags convey strong sentiment."""
    payload = {
        "texts": [
            "Worst customer support #WorstServiceEver #GhatiyaService",
            "Totally worth the price #PaisaVasool #MasterPieceUpdate",
            "Uninstalling immediately #BoycottApp"
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["label"] == "negative"
    assert results[1]["label"] == "positive"
    assert results[2]["label"] == "negative"

def test_2026_meme_templates(client):
    """Verifies 2026 viral meme and brainrot templates."""
    payload = {
        "texts": [
            "Bro thought he cooked 💀",
            "Who let him cook 💀",
            "Never let bro cook again",
            "He cooked fr 🔥",
            "Aura -1000",
            "Infinite aura 🏆",
            "Skill issue on the backend",
            "Nah bro this ain't it chief",
            "Very demure, very mindful"
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["label"] == "negative"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "negative"
    assert results[3]["label"] == "positive"
    assert results[4]["label"] == "negative"
    assert results[5]["label"] == "positive"
    assert results[6]["label"] == "negative"
    assert results[7]["label"] == "negative"
    assert results[8]["label"] == "positive"

def test_single_word_reaction_replies(client):
    """Verifies single-word or isolated reaction comments common on X, Reddit, and IG."""
    payload = {
        "texts": [
            "W",
            "L",
            "Ratio 💀",
            "Peak!",
            "Cooked 💀",
            "Fire 🔥"
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "negative"
    assert results[3]["label"] == "positive"
    assert results[4]["label"] == "negative"
    assert results[5]["label"] == "positive"

def test_hinglish_maranglish_phonetic_spellings(client):
    """Verifies phonetic variants in informal Indic social media text."""
    payload = {
        "texts": [
            "App khup chaan ahe, full paisa wasool!",
            "Ekdum bakwaas service hai, paise fukatt gele",
            "Bahut badiya app hai, maza aa gaya ekdum",
            "भावाने एकदम तोड काम केलंय, नादच खुळा!",
            "कतई जहर फीचर्स हैं इस नए अपडेट में, मजा आ गया!"
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "positive"
    assert results[3]["label"] == "positive"
    assert results[4]["label"] == "positive"
