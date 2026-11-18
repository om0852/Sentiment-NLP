"""Two-tier cascade (step 4), news handling (step 6) and calibrated confidence (step 5).

Tier 1: the TF-IDF model plus the kept hand-written rules. Microseconds per post.
Tier 2: the distilled multilingual transformer (src/models/student_onnx.py). Milliseconds per post.

A post goes to tier 2 when tier 1 is unlikely to be right:
  * tier-1 calibrated confidence is below `escalate_below`,
  * it reads like news or a factual headline (TF-IDF reacts to scary words, not to what happened),
  * it has sarcasm cues that bag-of-words cannot read, or
  * it is mostly in a non-Latin script (the TF-IDF vocabulary is mostly English/Hinglish).

If tier 2's calibrated confidence is still below `uncertain_below`, the post is flagged `uncertain`
so callers can route it to a human or an LLM instead of trusting the label.

Thresholds come from models/cascade_config.json, written by scripts/evaluate_cascade.py --write-config.
"""
import copy
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.calibration import softmax
from src.models.news_detector import NewsDetector

LABELS = ["negative", "neutral", "positive"]

DEFAULT_CONFIG = {
    "tfidf_temperature": 1.0,
    "escalate_below": 0.70,
    "uncertain_below": 0.55,
    "escalate_news": True,
    "news_threshold": 0.7,
    "escalate_sarcasm_cues": True,
    "escalate_non_latin": True,
    # short Latin-script social posts stay on tier 1, where the slang/emoji dictionaries and rules live;
    # measured: the student loses ~30 points on short Gen-Z/emoji/Hinglish posts but wins on longer ones
    "short_text_chars": 0,
    "keep_romanized_indic_on_tier1": False,
    "disabled_rules": [],
}

# ---------------- news / factual headline detector (step 6) ----------------
_REPORTING = re.compile(
    r"\b(says|said|reports?|reported|according to|announces?|announced|amid|slams?|hikes?|cuts?|kills?|killed|"
    r"injured|dies|died|arrested|launch(es|ed)?|unveils?|rises?|falls?|jumps?|surges?|plunges?|crash(es|ed)|"
    r"declares?|approves?|rejects?|probe|verdict|court|police|minister|government|govt|ministry|ceo|"
    r"quarter(ly)?|stocks?|sensex|nifty|rbi|imd|percent|per cent|crore|lakh|million|billion)\b",
    re.IGNORECASE,
)
_WIRE = re.compile(r"\((PTI|ANI|IANS|Reuters|AP|AFP)\)|\b(PTI|ANI|IANS|Reuters)\b|\|\s*[A-Z][\w ]+$")
# "us" only in lower case, so "US stocks surge" (United States) still counts as news
_FIRST_PERSON = re.compile(r"(?i:\b(i|i'm|im|i've|me|my|mine|we|we're|our|lol|lmao|ngl|tbh|fr|bro)\b)|\bus\b")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_NUMBER = re.compile(r"\d")


_NEWS_MODEL = None
_NEWS_MODEL_LOADED = False


def _news_model():
    """Learned detector from scripts/train_news_detector.py, if present."""
    global _NEWS_MODEL, _NEWS_MODEL_LOADED
    if not _NEWS_MODEL_LOADED:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "news_detector.npz")
        if NewsDetector.available(path):
            try:
                _NEWS_MODEL = NewsDetector(path)
            except Exception:
                _NEWS_MODEL = None
        _NEWS_MODEL_LOADED = True  # set only after the model is in place
    return _NEWS_MODEL


def news_probability(texts: List[str]) -> List[float]:
    m = _news_model()
    if m is None:
        return [1.0 if _heuristic_news(t) else 0.0 for t in texts]
    return m.proba(texts)


def is_news_like(text: str, threshold: float = 0.7) -> bool:
    if not text or _FIRST_PERSON.search(text) or len(_EMOJI.findall(text)) >= 2:
        return False
    return news_probability([text])[0] >= threshold


def _heuristic_news(text: str) -> bool:
    """Fallback when the learned detector is missing: third-person factual reporting."""
    if not text or len(text) < 25:
        return False
    if _FIRST_PERSON.search(text) or len(_EMOJI.findall(text)) >= 2 or text.count("!") >= 2:
        return False
    score = 0
    score += min(2, len(_REPORTING.findall(text)))
    score += 1 if _WIRE.search(text) else 0
    score += 1 if _NUMBER.search(text) else 0
    words = text.split()
    caps = sum(1 for w in words[:12] if w[:1].isupper())
    score += 1 if len(words) >= 5 and caps / min(12, len(words)) >= 0.5 else 0
    return score >= 3


# ---------------- sarcasm cues bag-of-words cannot read ----------------
_SARCASM_CUES = re.compile(
    r"(👏|🙃|🙂|🙄|😒|\b(love|loving) how\b|^\s*(oh\s+)?(great|wonderful|fantastic|perfect|yay|amazing)\b[,.!]|"
    r"\bthanks? for (absolutely )?nothing\b|\bcan'?t wait to\b|\bsure,? because\b|\bah yes\b|\bnothing says\b|"
    r"\bshout\s?out to\b|\btrue legend\b|\bwould .{0,30} again\b|\bsuch generosity\b|\bjust what i needed\b|"
    r"\bexactly (how|what) i wanted\b|\bcouldn'?t (be happier|ask for more)\b|\bliving (my|the) (best )?(life|dream)\b|"
    r"\b(totally|very|so) (normal|helpful|fun)\b.*\b(not|never|again|hours?|weeks?)\b)",
    re.IGNORECASE,
)


def has_sarcasm_cue(text: str) -> bool:
    return bool(_SARCASM_CUES.search(text or ""))


# Common romanized Hindi/Marathi function words. Latin-script Hinglish/Maranglish posts are where the
# transformer is weakest (its training data has little of it), and where the slang dictionaries cover most.
_ROMAN_INDIC = set("""
hai hain nahi nahin nhi kya kyu kyun bhai yaar yar ekdum khup ahe aahe aahes kar karo kiya diya gaya gayi mast
bahut bohot bhot kuch mein toh bhi ka ki ke ko ho raha rahi rahe paisa paise bhava bhau lai bhari chhan chan
kaam kam pura sab aaj kal hu hoon tha thi hota hoti zala jhala nako kay kaay mala tula tumhi aamhi apan
accha acha achha bakwas bekar faltu ghatiya jhakaas zabardast kamaal dimaag dimag watt laga vasool wala wali
""".split())
_WORD = re.compile(r"[a-z]+")


def is_romanized_indic(text: str) -> bool:
    words = _WORD.findall((text or "").lower())
    hits = sum(w in _ROMAN_INDIC for w in words)
    return hits >= 2 and hits / max(1, len(words)) >= 0.2


def non_latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters)


def load_config(path: Optional[str] = None) -> Dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "models", "cascade_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def calibrate_tfidf(probabilities: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Temperature-scale the TF-IDF model's probabilities (step 5)."""
    p = np.array([max(probabilities.get(l, 0.0), 1e-9) for l in LABELS])
    q = softmax(np.log(p)[None, :], temperature)[0]
    return {l: float(v) for l, v in zip(LABELS, q)}


CONTEXT_RULES = ("context_mismatch_sarcasm", "context_reinforced_negative", "context_aligned_praise")


class Cascade:
    def __init__(self, student=None, config: Optional[Dict] = None):
        self.student = student
        self.cfg = config or load_config()

    def escalation_reason(self, text: str, tier1_conf: float, tier1_reason: Optional[str],
                          news: bool = False) -> Optional[str]:
        """Why this post should go to tier 2, or None to keep the tier-1 answer."""
        if self.student is None:
            return None
        if tier1_reason in CONTEXT_RULES:
            return None  # thread context rules use information the student never sees
        if self.cfg.get("rules_override_student") and tier1_reason not in (None, "model_inference"):
            return None  # a hand-written rule decided; don't spend CPU on an answer we'd discard
        if self.cfg["escalate_news"] and news:
            return "news_like"
        if len(text) < self.cfg.get("short_text_chars", 0):
            return None  # short posts: slang/emoji dictionaries beat the student here
        if self.cfg.get("keep_romanized_indic_on_tier1") and is_romanized_indic(text):
            return None  # Hinglish/Maranglish in Latin script: same reason
        if self.cfg["escalate_sarcasm_cues"] and has_sarcasm_cue(text):
            return "sarcasm_cue"  # before the confidence check: bag-of-words is often confidently wrong here
        if tier1_conf >= self.cfg["escalate_below"]:
            return None  # tier 1 (calibrated) is very sure, e.g. dictionary-backed Hindi/Marathi
        if self.cfg["escalate_non_latin"] and non_latin_share(text) > 0.3:
            return "non_latin_script"
        return "low_confidence"

    def run_student(self, texts: List[str]) -> List[Dict]:
        return self.student.predict_batch(texts) if (self.student and texts) else []

    def is_uncertain(self, confidence: float) -> bool:
        return confidence < self.cfg["uncertain_below"]

    def describe(self) -> Tuple[bool, Dict]:
        return self.student is not None, {k: v for k, v in self.cfg.items() if k != "disabled_rules"}


def classify_batch(texts: List[str], contexts: List[Optional[str]], model, pipeline, context_analyzer,
                   cascade: Optional[Cascade]) -> List[Dict]:
    """Full decision for a batch of posts. Shared by the API and the evaluation scripts so that what we
    measure is exactly what we serve.

    Returns one dict per post with: label, confidence, probabilities, reason, tier ('tfidf'|'transformer'),
    content_type ('news'|'social'), uncertain, meta (pipeline flags), ctx_fallback.
    """
    from src.preprocessing.post_text import post_to_text

    texts = [t if isinstance(t, str) else "" for t in texts]
    processed, metas = [], []
    for t in texts:
        p, meta = pipeline.process(t)
        processed.append(p)
        metas.append(meta)
    preds = model.predict_batch(processed)

    cfg = cascade.cfg if cascade else DEFAULT_CONFIG
    temperature = cfg["tfidf_temperature"]
    clean = [post_to_text(t) or t for t in texts]
    # content_type is reported even when the transformer tier is off
    news_p = news_probability(clean)

    results, escalate = [], {}
    for i, (t, ctx, pred) in enumerate(zip(texts, contexts, preds)):
        cal = calibrate_tfidf(pred["probabilities"], temperature)
        base = max(cal, key=cal.get)
        label, conf, ctx_fallback, reason = context_analyzer.analyze(t, base, cal[base], context=ctx)
        is_news = (news_p[i] >= cfg.get("news_threshold", 0.7) and not _FIRST_PERSON.search(clean[i])
                   and len(_EMOJI.findall(clean[i])) < 2)
        results.append({
            "label": label, "confidence": conf, "probabilities": cal, "reason": reason,
            "tier": "tfidf", "content_type": "news" if is_news else "social", "uncertain": False,
            "meta": metas[i], "ctx_fallback": ctx_fallback, "tier1_label": label,
        })
        if cascade is not None and clean[i].strip():  # blank posts stay on tier 1

            why = cascade.escalation_reason(clean[i], conf, reason, news=is_news)
            if why:
                escalate[i] = why

    if escalate:
        idx = list(escalate)
        for i, s in zip(idx, cascade.run_student([clean[j] for j in idx])):
            r = results[i]
            if cascade.cfg.get("rules_override_student") and r["reason"] != "model_inference":
                continue
            r.update(label=s["label"], confidence=s["confidence"], probabilities=s["probabilities"],
                     reason=f"transformer_{escalate[i]}", tier="transformer", ctx_fallback=False,
                     uncertain=cascade.is_uncertain(s["confidence"]))
    return results
