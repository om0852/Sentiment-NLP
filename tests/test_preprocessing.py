import pytest
import sys
import os

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline

@pytest.fixture
def pipeline():
    return PreprocessingPipeline()

def test_slang_normalization(pipeline):
    text = "The new UI update is so fire no cap"
    processed, meta = pipeline.process(text)
    assert "amazing" in processed or "great" in processed
    assert meta["composite_polarity"] > 0

def test_negative_slang(pipeline):
    text = "Their customer support is mid and completely cooked"
    processed, meta = pipeline.process(text)
    assert "mediocre" in processed or "ruined" in processed
    assert meta["composite_polarity"] < 0

def test_emoji_handling(pipeline):
    text = "Bro this app crashed again 🤡"
    processed, meta = pipeline.process(text)
    assert "foolish" in processed or "clown" in processed
    assert meta["emoji_polarity"] < 0

def test_skull_humor_emoji(pipeline):
    text = "I am dying of laughter 💀"
    processed, meta = pipeline.process(text)
    assert "skull" in processed or "laughter" in processed or "hilarious" in processed
    assert meta["emoji_polarity"] > 0

def test_negation_scope(pipeline):
    text = "This product is not good at all"
    processed, meta = pipeline.process(text)
    assert "not_good" in processed

def test_sarcasm_marker_detection(pipeline):
    text = "Oh wow, what a genius move /s"
    processed, meta = pipeline.process(text)
    assert meta["sarcasm_detected"] is True

def test_hashtag_splitting(pipeline):
    text = "#WorstCustomerService ever"
    processed, meta = pipeline.process(text)
    assert "worst" in processed
    assert "customer" in processed
    assert "service" in processed
