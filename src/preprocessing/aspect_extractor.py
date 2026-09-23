import re
from typing import Dict, List, Optional, Tuple

ASPECT_KEYWORDS = {
    "ui_ux": [
        "ui", "ux", "design", "look", "looks", "clean", "theme", "aesthetic",
        "button", "buttons", "layout", "animation", "animations", "screen",
        "interface", "visual", "visuals", "dark mode", "font", "colors"
    ],
    "performance": [
        "speed", "fast", "slow", "slower", "lag", "lagging", "smooth", "smoother",
        "crash", "crashes", "crashed", "crashing", "freeze", "freezes", "frozen",
        "bug", "bugs", "buggy", "glitch", "glitches", "memory", "battery",
        "cpu", "watt laga di", "load time", "loading", "optimize", "optimized"
    ],
    "customer_support": [
        "support", "service", "customer support", "customer service", "agent",
        "ticket", "tickets", "response", "refund", "refunds", "email", "helpdesk",
        "live chat", "ghosted", "call", "representative"
    ],
    "pricing_value": [
        "price", "pricing", "cost", "subscription", "cheap", "expensive",
        "ripoff", "rip off", "rip-off", "scam", "paisa vasool", "paisa wasool",
        "worth it", "waste of money", "loot liya", "chindi", "paywall", "affordable"
    ],
    "features": [
        "feature", "features", "update", "patch", "tool", "tools", "mode",
        "matchmaking", "export", "dashboard", "functionality", "weapons", "release"
    ]
}

POSITIVE_SIGNALS = {
    "fire", "slaps", "clean", "smooth", "fast", "goated", "w", "huge w", "love",
    "amazing", "great", "good", "perfect", "chef's kiss", "immaculate", "op",
    "mast", "ekdum mast", "cracked", "bawaal", "jhakaas", "paisa vasool", "solid",
    "awesome", "helpful", "responsive", "quick", "affordable", "worth", "worth it"
}

NEGATIVE_SIGNALS = {
    "trash", "garbage", "mid", "cooked", "crash", "crashes", "slow", "slower",
    "bug", "bugs", "buggy", "l", "huge l", "flop", "scam", "hate", "terrible",
    "worst", "broken", "bricked", "annoying", "cringe", "confusing", "fumble",
    "watt laga di", "dimaag kharab", "loot liya", "stupid", "ripoff", "rip-off",
    "rip off", "ghosted", "unresponsive", "waste", "useless", "overpriced", "fails",
    "failed", "fail", "glitch", "glitches", "lag", "lagging", "chindi"
}

class AspectExtractor:
    def __init__(self):
        # Precompile regex boundaries for aspects
        self.aspect_patterns: Dict[str, List[re.Pattern]] = {}
        for aspect, kws in ASPECT_KEYWORDS.items():
            patterns = []
            for kw in kws:
                patterns.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
            self.aspect_patterns[aspect] = patterns

    def extract_aspects(self, text: str) -> Dict[str, str]:
        """
        Extracts mentioned aspects from text and determines per-aspect sentiment.
        Returns e.g.: {'ui_ux': 'positive', 'performance': 'negative'}
        """
        if not text or not isinstance(text, str):
            return {}

        text_lower = text.lower()
        # Split text into logical clauses by punctuation or contrastive words
        clauses = re.split(r"[,;.\n]|(?:\s+(?:but|however|although|yet|and|while|dusri side|ek side)\s+)", text_lower)
        
        aspect_results: Dict[str, str] = {}

        for aspect, patterns in self.aspect_patterns.items():
            # Check which clauses mention this aspect
            relevant_clauses = []
            for clause in clauses:
                for pat in patterns:
                    if pat.search(clause):
                        relevant_clauses.append(clause)
                        break

            if relevant_clauses:
                # Score the sentiment of the clauses mentioning this aspect
                pos_count = 0
                neg_count = 0
                for clause in relevant_clauses:
                    # Check multi-word signals first
                    for sig in POSITIVE_SIGNALS:
                        if " " in sig and sig in clause:
                            pos_count += 2
                    for sig in NEGATIVE_SIGNALS:
                        if " " in sig and sig in clause:
                            neg_count += 2

                    words = set(re.findall(r"\b[\w'-]+\b", clause))
                    pos_count += len(words.intersection(POSITIVE_SIGNALS))
                    neg_count += len(words.intersection(NEGATIVE_SIGNALS))

                if pos_count > neg_count:
                    aspect_results[aspect] = "positive"
                elif neg_count > pos_count:
                    aspect_results[aspect] = "negative"
                else:
                    aspect_results[aspect] = "neutral"

        # Invert positive aspects if a sarcastic punchline reversal / lie is present
        lie_inversion = bool(re.search(r'(?<!not\s)(?<!not\sa\s)\b(what\s+(they|i|we)?\s*say\s+is\s+a\s+lie|what\s+say\s+is\s+a\s+lie|what\s+they\s+say\s+is\s+cap|that\s+was\s+a\s+lie|is\s+a\s+lie|was\s+a\s+lie|said\s+no\s+one\s+ever|in\s+my\s+dreams|psych\b|sike\b)\b', text_lower))
        if lie_inversion:
            for aspect, val in aspect_results.items():
                if val == "positive":
                    aspect_results[aspect] = "negative"

        return aspect_results
