import numpy as np
import pytest

from src.models.calibration import expected_calibration_error, fit_temperature, softmax
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.preprocessing.post_text import post_to_text
from src.serving.cascade import (
    DEFAULT_CONFIG,
    Cascade,
    calibrate_tfidf,
    classify_batch,
    has_sarcasm_cue,
    non_latin_share,
)


# ---------------- post_to_text ----------------
def test_post_to_text_strips_html_entities_and_platform_suffix():
    out = post_to_text("<b>Canva</b> tips &amp; tricks - Instagram", None)
    assert out == "Canva tips & tricks"


def test_post_to_text_drops_description_repeating_title():
    t = "Blessed beyond measure #newyear"
    assert post_to_text(t, t) == t


def test_post_to_text_removes_photo_boilerplate():
    out = post_to_text("New logo drop", "Photo by Arnab | Designer on May 14, 2025.")
    assert "Photo by" not in out and out.startswith("New logo drop")


# ---------------- calibration ----------------
def test_fit_temperature_softens_overconfident_logits():
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 3, size=2000)
    logits = rng.normal(size=(2000, 3))
    # make the model right 60% of the time but extremely confident
    right = rng.rand(2000) < 0.6
    for i in range(2000):
        k = labels[i] if right[i] else (labels[i] + 1) % 3
        logits[i, k] += 12.0
    t = fit_temperature(logits, labels)
    assert t > 2.0
    before = expected_calibration_error(softmax(logits), labels)
    after = expected_calibration_error(softmax(logits, t), labels)
    assert after < before


def test_calibrate_tfidf_keeps_argmax():
    probs = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    cal = calibrate_tfidf(probs, 2.0)
    assert max(cal, key=cal.get) == "positive"
    assert cal["positive"] < 0.7
    assert abs(sum(cal.values()) - 1.0) < 1e-6


# ---------------- routing signals ----------------
@pytest.mark.parametrize("text", [
    "Love how my internet dies right before a deadline",
    "Great, the one day I forget my umbrella",
    "thanks for nothing 🙂",
])
def test_sarcasm_cues(text):
    assert has_sarcasm_cue(text)


def test_no_sarcasm_cue_on_plain_praise():
    assert not has_sarcasm_cue("The battery lasts two full days, very happy")


def test_first_person_filter_ignores_us_country():
    from src.serving.cascade import _FIRST_PERSON
    assert not _FIRST_PERSON.search("US stocks surge as inflation cools")
    assert _FIRST_PERSON.search("join us tonight")
    assert _FIRST_PERSON.search("I love it")
    assert not _FIRST_PERSON.search("Imports rise in October")


def test_non_latin_share():
    assert non_latin_share("खूप छान अनुभव") > 0.9
    assert non_latin_share("bhai ye phone mast hai") == 0.0


# ---------------- cascade ----------------
class FakeStudent:
    def __init__(self, label="negative", confidence=0.9):
        self.label, self.confidence = label, confidence
        self.seen = []

    def predict_batch(self, texts):
        self.seen.extend(texts)
        probs = {"negative": 0.05, "neutral": 0.05, "positive": 0.05}
        probs[self.label] = self.confidence
        return [{"label": self.label, "confidence": self.confidence, "probabilities": dict(probs)} for _ in texts]


@pytest.fixture(scope="module")
def tier1():
    from src.models import PureSentimentClassifier
    from src.preprocessing.pipeline import PreprocessingPipeline
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = PureSentimentClassifier()
    model.load(os.path.join(root, "models", "sentiment_model.json.gz"))
    return model, PreprocessingPipeline(), ContextAnalyzer(disabled_rules=[])


def test_escalation_reasons():
    c = Cascade(student=FakeStudent(), config=dict(DEFAULT_CONFIG, escalate_below=0.7))
    assert c.escalation_reason("anything", 0.99, "model_inference", news=True) == "news_like"
    assert c.escalation_reason("Yay, Monday", 0.99, "model_inference") == "sarcasm_cue"
    assert c.escalation_reason("खूप छान अनुभव", 0.5, "model_inference") == "non_latin_script"
    assert c.escalation_reason("खूप छान अनुभव", 0.99, "model_inference") is None  # confident tier 1 keeps it
    assert c.escalation_reason("nice phone", 0.5, "model_inference") == "low_confidence"
    assert c.escalation_reason("nice phone", 0.99, "model_inference") is None
    # thread-context rules see information the student never gets, so they are never overridden
    assert c.escalation_reason("Brilliant job 👏", 0.1, "context_mismatch_sarcasm") is None


def test_fired_rule_skips_student_when_rules_override():
    cfg = dict(DEFAULT_CONFIG, escalate_below=1.01, rules_override_student=True)
    c = Cascade(student=FakeStudent(), config=cfg)
    assert c.escalation_reason("Who let him cook, honestly the worst update", 0.95, "meme_mockery_cooked") is None
    assert c.escalation_reason("Who let him cook, honestly the worst update", 0.95, "model_inference") == "low_confidence"


def test_short_and_romanized_posts_stay_on_tier1():
    cfg = dict(DEFAULT_CONFIG, escalate_below=1.01, short_text_chars=60, keep_romanized_indic_on_tier1=True)
    c = Cascade(student=FakeStudent(), config=cfg)
    assert c.escalation_reason("this fit is bussin no cap", 0.5, "model_inference") is None
    assert c.escalation_reason("Bhai kya update diya hai, pura dimaag kharab kar diya, watt laga di yaar", 0.5,
                               "model_inference") is None
    assert c.escalation_reason("Sensex", 0.5, "model_inference", news=True) == "news_like"
    long_post = "The conference was well organized and the talks were excellent, networking felt rushed"
    assert c.escalation_reason(long_post, 0.99, "model_inference") == "low_confidence"


def test_no_student_means_tfidf_only(tier1):
    model, pipe, analyzer = tier1
    res = classify_batch(["Yay, Monday"], [None], model, pipe, analyzer, Cascade(student=None))
    assert res[0]["tier"] == "tfidf" and res[0]["uncertain"] is False


def test_blank_posts_never_escalate(tier1):
    model, pipe, analyzer = tier1
    student = FakeStudent()
    cfg = dict(DEFAULT_CONFIG, escalate_below=1.01)
    res = classify_batch(["   "], [None], model, pipe, analyzer, Cascade(student=student, config=cfg))
    assert res[0]["tier"] == "tfidf" and student.seen == []


def test_student_decides_escalated_posts_and_flags_uncertain(tier1):
    model, pipe, analyzer = tier1
    student = FakeStudent(label="negative", confidence=0.4)
    cfg = dict(DEFAULT_CONFIG, escalate_below=0.0, uncertain_below=0.5, escalate_news=False)
    res = classify_batch(["Yay, Monday", "this phone is great"], [None, None], model, pipe, analyzer,
                         Cascade(student=student, config=cfg))
    assert res[0]["tier"] == "transformer" and res[0]["label"] == "negative"
    assert res[0]["reason"] == "transformer_sarcasm_cue" and res[0]["uncertain"] is True
    assert res[1]["tier"] == "tfidf"
    assert student.seen == ["Yay, Monday"]


def test_disabled_rule_falls_back_to_model():
    text = "this update is a skill issue"
    on = ContextAnalyzer(disabled_rules=[]).analyze(text, "positive", 0.8)
    off = ContextAnalyzer(disabled_rules=["meme_skill_issue"]).analyze(text, "positive", 0.8)
    assert on[3] == "meme_skill_issue" and on[0] == "negative"
    assert off[3] == "model_inference" and off[0] == "positive"


def test_slang_dictionary_answer_is_kept():
    c = Cascade(student=FakeStudent(), config=dict(DEFAULT_CONFIG, slang_keep_conf=0.6))
    meta_pos = {"slang_polarity": 1.8}
    # "op" (overpowered) recognised by the slang dictionary, tier 1 agrees and is confident -> keep tier 1
    assert c.slang_backed("nothing can be this much op", "positive", 0.7, "model_inference", meta_pos, news=False)
    # not when tier 1 disagrees with the slang direction, is unsure, the post is news, or has sarcasm cues
    assert not c.slang_backed("nothing can be this much op", "negative", 0.9, "model_inference", meta_pos, news=False)
    assert not c.slang_backed("nothing can be this much op", "positive", 0.5, "model_inference", meta_pos, news=False)
    assert not c.slang_backed("Fire broke out in a factory", "positive", 0.9, "model_inference", meta_pos, news=True)
    assert not c.slang_backed("Great, the app is fire 🙃", "positive", 0.9, "model_inference", meta_pos, news=False)
    # off when not configured
    off = Cascade(student=FakeStudent(), config=dict(DEFAULT_CONFIG))
    assert not off.slang_backed("op bro", "positive", 0.99, "model_inference", meta_pos, news=False)


def test_op_is_positive_end_to_end(tier1):
    model, pipe, analyzer = tier1
    cfg = dict(DEFAULT_CONFIG, slang_keep_conf=0.6, escalate_below=1.01)
    res = classify_batch(["nothing can be this much op"], [None], model, pipe, analyzer,
                         Cascade(student=FakeStudent(label="neutral", confidence=0.9), config=cfg))
    assert res[0]["label"] == "positive" and res[0]["reason"] == "slang_dictionary"
