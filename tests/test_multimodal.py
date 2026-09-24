import io
import pytest
from PIL import Image, ImageDraw
from starlette.testclient import TestClient
from src.serving.app import app
from src.multimodal import MultimodalPipeline, DomainCategorizer, SemanticTagger

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def create_sample_image(text: str) -> bytes:
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 50), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def test_analyze_image_endpoint(client):
    img_bytes = create_sample_image("Lai bhari update ahe bhava! Full paisa vasool #BestApp")
    files = {"file": ("test_praise.png", img_bytes, "image/png")}
    
    resp = client.post("/analyze/image", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["media_type"] == "image"
    assert data["file_format"] == "png"
    assert "sentiment" in data
    assert data["sentiment"]["label"] in ("positive", "negative", "neutral")
    assert "category" in data
    assert isinstance(data["tags"], list)
    assert len(data["tags"]) > 0
    assert data["metadata"]["width"] == 600
    assert data["metadata"]["height"] == 150

def test_analyze_document_endpoint(client):
    content = "Critical memory leak in the billing webhook crashed our cluster. Error 500 continuously."
    files = {"file": ("incident_report.txt", content.encode("utf-8"), "text/plain")}
    
    resp = client.post("/analyze/document", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["media_type"] == "document"
    assert data["sentiment"]["label"] == "negative"
    assert data["category"] == "Tech & Software Bugs"
    assert any("leak" in t or "crash" in t or "tech" in t for t in data["tags"])

def test_analyze_universal_file_endpoint(client):
    content = "This press release contains forward-looking financial statements. The board of directors held an annual meeting."
    files = {"file": ("quarterly_announcement.txt", content.encode("utf-8"), "text/plain")}
    
    resp = client.post("/analyze/file", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["media_type"] == "document"
    assert data["sentiment"]["label"] == "neutral"
    assert data["category"] == "News & Corporate Announcement"
    assert any("news" in t or "announcement" in t or "press" in t for t in data["tags"])

def test_domain_categorizer_and_tagger_direct():
    categorizer = DomainCategorizer()
    tagger = SemanticTagger()

    # 1. Meme
    meme_text = "Bro thought he cooked 💀 infinite aura -1000 #AuraLoss"
    cat_res = categorizer.classify(meme_text)
    tags = tagger.extract_tags(meme_text, category=cat_res["category"], sentiment_label="negative")
    assert cat_res["category"] == "Meme / Social Humor"
    assert any("meme" in t or "aura" in t for t in tags)

    # 2. Finance & Payment
    finance_text = "Transaction failed, money debited but ticket not booked! Fraud bank service."
    cat_res = categorizer.classify(finance_text)
    tags = tagger.extract_tags(finance_text, category=cat_res["category"], sentiment_label="negative")
    assert cat_res["category"] == "Finance & Payment Issue"
    assert any("payment" in t or "money" in t or "finance" in t for t in tags)

    # 3. Product review
    praise_text = "App khup chhan ahe, full paisa vasool! Ek number update."
    cat_res = categorizer.classify(praise_text)
    tags = tagger.extract_tags(praise_text, category=cat_res["category"], sentiment_label="positive")
    assert cat_res["category"] == "Product & E-Commerce Review"
    assert any("marathi_praise" in t or "value_for_money" in t or "product" in t for t in tags)

def test_multimodal_pipeline_direct():
    pipeline = MultimodalPipeline()
    res = pipeline.analyze("Zero bugs, zero downtime, and our latency dropped by 80%.", filename="benchmark_notes.txt")
    
    assert res["sentiment"]["label"] == "positive"
    assert res["sentiment"]["confidence"] >= 0.80
    assert isinstance(res["tags"], list)
    assert res["latency_ms"] >= 0.0

def test_payment_failure_ui_detection():
    # Test that payment UI modal with technical issue is recognized as Finance & Payment Issue
    categorizer = DomainCategorizer()
    tagger = SemanticTagger()
    meta = {"is_payment_ui": True, "detected_gateway": "PhonePe / UPI Theme"}
    cat_res = categorizer.classify("Technical Issue", metadata=meta)
    tags = tagger.extract_tags("Technical Issue", category=cat_res["category"], sentiment_label="negative", metadata=meta)
    
    assert cat_res["category"] == "Finance & Payment Issue"
    assert "Payment Failure" in cat_res["subcategory"]
    assert any("payment" in t or "upi" in t or "phonepe" in t for t in tags)
