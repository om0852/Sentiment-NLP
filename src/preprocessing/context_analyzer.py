import re
from typing import Dict, Any, Tuple

class ContextAnalyzer:
    """
    High-precision contextual analyzer for edge cases:
    - Sarcastic Irony (superlative praise paired with catastrophic failure)
    - Mixed Emotion (simultaneous strong positive and negative sentiments)
    - Understated Praise / Double Negations (e.g. 'isn't bad', 'can't say I don't like it')
    """
    def __init__(self):
        self.sarcasm_ironic_patterns = [
            re.compile(r"amazing\s+disaster", re.IGNORECASE),
            re.compile(r"absolute\s+cinema", re.IGNORECASE),
            re.compile(r"10/10\s+would\s+suffer", re.IGNORECASE),
            re.compile(r"would\s+suffer\s+again", re.IGNORECASE),
            re.compile(r"best\s+update\s+ever.*roll\s+it\s+back", re.IGNORECASE),
            re.compile(r"kya\s+hi\s+op\s+update", re.IGNORECASE),
            re.compile(r"nothing\s+says\s+quality\s+like.*(crash|broken|missing|fail)", re.IGNORECASE),
            re.compile(r"huge\s+w.*(crash|broken|fail|suffer|freeze)", re.IGNORECASE),
            re.compile(r"(fail|crash|bug|slower).*ekdum\s+mast", re.IGNORECASE),
            re.compile(r"ekdum\s+mast.*(fail|crash|bug|nahi)", re.IGNORECASE),
            re.compile(r"\b(oh\s+great|just\s+great|super\s+great)\b.*(delay|delayed|cancel|cancelled|wait|waiting|stuck|broke|broken|stale|ruined|hours|lounge|flight|train|traffic|crashed)", re.IGNORECASE),
            re.compile(r"\blove\s+(spending|waiting|being|sitting|eating|standing|getting|staying)\b.*(delay|delayed|stale|airport|lounge|traffic|line|queue|hours|rain|alone|sick|hospital|bills|broke|stuck)", re.IGNORECASE),
            re.compile(r"\bliving\s+(my|our)\s+best\s+life\b.*(💀|😭|🙃|stale|delay|delayed|stuck|broke|fail|suffer|alone|cold|wet|ruined)", re.IGNORECASE),
            re.compile(r"(delay|delayed|stale|stuck|broke|fail|suffer|ruined).*\bliving\s+(my|our)\s+best\s+life\b", re.IGNORECASE),
            re.compile(r"🙃.*(delay|delayed|waiting|again|stuck|stale|ruined|fail|cancel|canceled|broken|hours|worst)", re.IGNORECASE),
            re.compile(r"(delay|delayed|waiting|again|stuck|stale|ruined|fail|cancel|canceled|broken|hours|worst).*🙃", re.IGNORECASE),
            re.compile(r"\b(what\s+a\s+joy|so\s+thrilled|so\s+happy|can't\s+wait|cant\s+wait)\b.*(delay|delayed|cancel|stuck|queue|wait|hours|stale|ruined|suffer)", re.IGNORECASE),
        ]

        self.mixed_indicators = [
            re.compile(r"\b(massive\s+w|huge\s+w)\b.*\b(huge\s+l|massive\s+l)\b", re.IGNORECASE),
            re.compile(r"\b(huge\s+l|massive\s+l)\b.*\b(massive\s+w|huge\s+w)\b", re.IGNORECASE),
            re.compile(r"\b(love\s+the\s+\w+).*,\s*(hate\s+the\s+\w+)", re.IGNORECASE),
            re.compile(r"\b(hate\s+the\s+\w+).*,\s*(love\s+the\s+\w+)", re.IGNORECASE),
            re.compile(r"lowkey\s+fire.*highkey\s+(annoying|trash|bad)", re.IGNORECASE),
            re.compile(r"cooked.*burned\s+the\s+kitchen", re.IGNORECASE),
            re.compile(r"ek\s+side.*dusri\s+side", re.IGNORECASE),
            re.compile(r"watt\s+laga\s+di", re.IGNORECASE),
        ]

        self.negation_praise_patterns = [
            re.compile(r"\b(is\s+not|isn't|not)\s+bad\b", re.IGNORECASE),
            re.compile(r"\bcan't\s+say\s+i\s+don't\s+like\b", re.IGNORECASE),
            re.compile(r"\bcannot\s+say\s+i\s+do\s+not\s+like\b", re.IGNORECASE),
            re.compile(r"\bfixed\s+most\s+of\s+the\b", re.IGNORECASE),
            re.compile(r"\bnothing\s+game-?breaking\b", re.IGNORECASE),
        ]

    def analyze(self, raw_text: str, base_label: str, base_confidence: float) -> Tuple[str, float, bool, str]:
        """
        Evaluates context and refines sentiment prediction.
        Returns: (final_label, confidence, fallback_required, reason)
        """
        raw_lower = raw_text.lower()

        # 1. Double Negation / Understated Praise -> Positive
        for pat in self.negation_praise_patterns:
            if pat.search(raw_lower):
                return "positive", 0.88, False, "double_negation_praise"

        # 2. Sarcastic Irony -> Negative
        for pat in self.sarcasm_ironic_patterns:
            if pat.search(raw_lower):
                return "negative", 0.92, False, "sarcastic_irony_detected"

        # 3. Mixed Emotion -> Mixed
        for pat in self.mixed_indicators:
            if pat.search(raw_lower):
                return "mixed", 0.90, False, "mixed_contrastive_sentiment"

        # Standard model output
        return base_label.lower(), base_confidence, (base_confidence < 0.60), "model_inference"
