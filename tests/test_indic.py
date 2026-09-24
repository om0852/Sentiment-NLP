import pytest
from starlette.testclient import TestClient
from src.serving.app import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_devanagari_hindi_sentiment(client):
    payload = {
        "texts": [
            "यह ऐप बहुत अच्छा है, एकदम मस्त!",
            "बिल्कुल घटिया सर्विस है, कोई मदद नहीं मिली।",
            "ऐप में बहुत सारे बग्स हैं और यह बार-बार क्रैश हो रहा है।"
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]

    # 1. Hindi Praise
    assert results[0]["label"] == "positive"
    assert results[0]["confidence"] >= 0.70

    # 2. Hindi Complaint
    assert results[1]["label"] == "negative"
    assert results[1]["confidence"] >= 0.70

    # 3. Hindi Crash
    assert results[2]["label"] == "negative"
    assert results[2]["is_performance_issue"] is True

def test_devanagari_marathi_sentiment(client):
    payload = {
        "texts": [
            "हा ॲप खूप छान आहे, मस्त काम करतो!",
            "एकदम बकवास ॲप, पैसे वाया गेले!",
            "काही कामाचा नाही, लगेच बंद पडतो."
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]

    # 1. Marathi Praise
    assert results[0]["label"] == "positive"
    assert results[0]["confidence"] >= 0.80

    # 2. Marathi Complaint
    assert results[1]["label"] == "negative"
    assert results[1]["confidence"] >= 0.80

    # 3. Marathi Bug/Crash
    assert results[2]["label"] == "negative"
    assert results[2]["is_performance_issue"] is True

def test_maranglish_and_hinglish_slang(client):
    payload = {
        "texts": [
            "App khup chhan ahe, ek number kam kela!",
            "Kahi upyog nahi, ekdum faltu ahe, paise fukat gele.",
            "Lai bhari update ahe bhava!",
            "Ye app ekdum mast hai, full paisa vasool!",
            "Bhai kya update diya hai, pura dimaag kharab kar diya, watt laga di."
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]

    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "positive"
    assert results[3]["label"] == "positive"
    assert results[4]["label"] == "negative"

def test_indic_negation_flipping(client):
    payload = {
        "texts": [
            "हा ॲप चांगला नाही",
            "ॲपमधील सर्व बग्स मिटले आहेत आणि आता समस्या नाही"
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]

    # 'चांगला नाही' -> should be negative
    assert results[0]["label"] == "negative"

    # 'बग्स मिटले... समस्या नाही' -> should be positive
    assert results[1]["label"] == "positive"

def test_indic_aspect_based_sentiment(client):
    payload = {
        "texts": [
            "हा ॲप दिखने में खूप छान आहे, पण स्पीड खूप स्लो आहे."
        ],
        "extract_aspects": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    item = results[0]

    assert "ui_ux" in item["aspects"]
    assert item["aspects"]["ui_ux"] == "positive"
    assert "performance" in item["aspects"]
    assert item["aspects"]["performance"] == "negative"
    assert item["is_mixed"] is True

def test_indic_double_meaning_and_sarcasm(client):
    payload = {
        "texts": [
            "Wah kya update diya hai, app khulte hi phone restart ho gaya!",
            "लय भारी काम केलं राव, पैसे कट झाले पण तिकीट मिळालंच नाही!",
            "दिसण्यात १ नंबर आहे, फक्त चालत नाही एवढंच.",
            "Best app ever if you love losing your money 💀",
            "पैसे बुडवल्याबद्दल मनापासून धन्यवाद!",
            "डेव्हलपर्स झोपले होते का हा अपडेट देताना?",
            "गाणं एकदम जहर आहे भावा!"
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]

    # 1. Sarcastic praise -> negative
    assert results[0]["label"] == "negative"
    # 2. Marathi Sarcastic praise -> negative
    assert results[1]["label"] == "negative"
    # 3. Backhanded compliment -> negative
    assert results[2]["label"] == "negative"
    # 4. Cynical conditional irony -> negative
    assert results[3]["label"] == "negative"
    # 5. Faux gratitude -> negative
    assert results[4]["label"] == "negative"
    # 6. Rhetorical mockery -> negative
    assert results[5]["label"] == "negative"
    # 7. Slang inversion praise -> positive
    assert results[6]["label"] == "positive"
