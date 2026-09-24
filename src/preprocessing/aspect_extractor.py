import re
from typing import Dict, List, Optional, Tuple

ASPECT_KEYWORDS = {
    "ui_ux": [
        "ui", "ux", "design", "look", "looks", "clean", "theme", "aesthetic", "aesthetics",
        "button", "buttons", "layout", "animation", "animations", "screen", "screens",
        "interface", "visual", "visuals", "dark mode", "font", "fonts", "colors",
        "typography", "visual masterpiece", "micro-interactions", "contrast", "palette", "canvas",
        # Indic (Hindi/Marathi)
        "दिसतो", "दिसते", "दिखने में", "स्क्रीन", "थीम", "बटण", "बटन", "फॉन्ट", "डिझाइन", "डिजाइन", "लुक", "रंग"
    ],
    "performance": [
        "speed", "fast", "slow", "slower", "lag", "lagging", "smooth", "smoother",
        "crash", "crashes", "crashed", "crashing", "freeze", "freezes", "frozen",
        "bug", "bugs", "buggy", "glitch", "glitches", "memory", "battery",
        "cpu", "watt laga di", "load time", "loading", "optimize", "optimized",
        "latency", "query latency", "backend query", "throughput", "overheating",
        "thermal throttling", "kernel panic", "space heater", "packet loss", "frame drops",
        "netcode", "audio compression", "transcription speed", "bottleneck",
        "timing out", "timeout", "timeouts", "database", "connection pool",
        # Indic (Hindi/Marathi)
        "स्पीड", "हँग", "हॅंग", "क्रैश", "स्लो", "गती", "फास्ट", "बग", "बग्स", "बंद पडतो", "बंद पडतोय", "चालत नाही", "खराब चालतो"
    ],
    "customer_support": [
        "support", "service", "customer support", "customer service", "agent", "agents",
        "ticket", "tickets", "response", "refund", "refunds", "email", "helpdesk",
        "live chat", "ghosted", "call", "representative", "customer success",
        "tier-one", "tier-two", "support desk", "escalation", "help desk",
        # Indic (Hindi/Marathi)
        "मदत", "सपोर्ट", "कॉल", "सेवा", "सर्व्हिस", "ग्राहक सेवा", "उत्तर", "मदत मिळत नाही"
    ],
    "pricing_value": [
        "price", "pricing", "cost", "subscription", "cheap", "expensive",
        "ripoff", "rip off", "rip-off", "scam", "paisa vasool", "paisa wasool",
        "worth it", "waste of money", "loot liya", "chindi", "paywall", "affordable",
        "shareholder value", "licensing", "annual renewal", "price hikes", "auto-renewed",
        "annual contract", "billing", "charges", "charged",
        # Indic (Hindi/Marathi)
        "पैसे", "किंमत", "महाग", "खर्च", "स्वस्त", "लूट", "सबस्क्रिप्शन", "पैसे वाया", "पैसे फुकट"
    ],
    "features": [
        "feature", "features", "update", "patch", "tool", "tools", "mode",
        "matchmaking", "export", "exports", "dashboard", "functionality", "weapons", "release",
        "sdk", "sdks", "driver", "webhook", "webhooks", "search index", "filter", "modal",
        "reporting", "integration", "integrations", "documentation", "api",
        # Indic (Hindi/Marathi)
        "फीचर", "फीचर्स", "अपडेट", "नवीन", "साधन", "टूल"
    ]
}

POSITIVE_SIGNALS = {
    "fire", "slap", "slaps", "clean", "smooth", "fast", "goated", "w", "huge w", "love",
    "amazing", "great", "good", "perfect", "chef's kiss", "immaculate", "op",
    "mast", "ekdum mast", "cracked", "bawaal", "jhakaas", "paisa vasool", "solid",
    "awesome", "helpful", "responsive", "quick", "affordable", "worth", "worth it",
    "masterpiece", "breathtaking", "world-class", "stunning", "undeniable", "flawlessly",
    "wicked", "sick", "magic", "snappy", "generous", "tactile", "clickiness",
    "ergonomic", "pristine", "studio-grade", "unmatched", "brilliant", "delight", "joy",
    # Marathi & Hindi colloquialisms
    "lai bhari", "lay bhari", "khup chhan", "ek number", "kadak", "bhari", "shandar", "badhiya",
    # Devanagari positive
    "छान", "मस्त", "उत्तम", "सुंदर", "बढ़िया", "शानदार", "लाजवाब", "झकास", "कडक", "भारी",
    "चांगला", "चांगली", "चांगले", "सोपा", "सोपी", "सोपे", "सुरक्षित", "सर्वोत्तम"
}

NEGATIVE_SIGNALS = {
    "trash", "garbage", "mid", "cooked", "crash", "crashes", "slow", "slower",
    "bug", "bugs", "buggy", "l", "huge l", "flop", "scam", "hate", "terrible",
    "worst", "broken", "bricked", "annoying", "cringe", "confusing", "fumble",
    "watt laga di", "dimaag kharab", "loot liya", "stupid", "ripoff", "rip-off",
    "rip off", "ghosted", "unresponsive", "waste", "useless", "overpriced", "fails",
    "failed", "fail", "glitch", "glitches", "lag", "lagging", "chindi",
    "nightmare", "unbearable", "bloatware", "infested", "throttling", "overheating",
    "drains", "erroneously", "predatory", "crippled", "clunky", "unoptimized",
    "mediocre", "dull", "cash grab", "dial-up", "unforgivable", "corrupted",
    "crashed", "crashing", "leak", "leaks", "leaked", "critical",
    "timeout", "timeouts", "timing out", "exhausted", "failing",
    # Marathi & Hindi colloquialisms
    "paise fukat", "paise vaya", "kahi upyog nahi", "band padto", "chalat nahi", "doke dukhi", "faltu", "bakwas",
    # Devanagari negative
    "घटिया", "बकवास", "बेकार", "खराब", "फालतू", "कचरा", "रद्दी", "धोखा", "लूट", "त्रास",
    "बग", "बग्स", "क्रैश", "हँग", "स्लो", "हळू", "महाग", "बंद पडतो", "चालत नाही", "वाईट"
}

class AspectExtractor:
    def __init__(self):
        self.aspect_patterns: Dict[str, List[re.Pattern]] = {}
        self.aspect_keyword_sets: Dict[str, set] = {}
        self.all_keywords: set = set()

        for aspect, kws in ASPECT_KEYWORDS.items():
            patterns = []
            kw_set = set(kws)
            self.aspect_keyword_sets[aspect] = kw_set
            self.all_keywords.update(kw_set)
            for kw in kws:
                if re.search(r"[\u0900-\u097F]", kw):
                    patterns.append(re.compile(rf"(?<![\w\u0900-\u097F]){re.escape(kw)}(?![\w\u0900-\u097F])", re.IGNORECASE))
                else:
                    patterns.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
            self.aspect_patterns[aspect] = patterns

    def extract_aspects(self, text: str) -> Dict[str, str]:
        """
        Extracts mentioned aspects from text and determines per-aspect sentiment.
        Returns a dict mapping aspect_name -> 'positive' | 'negative' | 'neutral'.
        """
        text_lower = text.lower()
        if not any(k in text_lower for k in self.all_keywords):
            return {}

        extracted: Dict[str, str] = {}

        # Sarcastic lie reversal guard
        is_sarcastic_reversal = bool(re.search(
            r"(?<!not\s)(?<!not\sa\s)\b(what\s+(they|i|we)?\s*say\s+is\s+a\s+lie|what\s+say\s+is\s+a\s+lie|what\s+they\s+say\s+is\s+cap|that\s+was\s+a\s+lie|turns\s+out\s+that\s+was\s+a\s+lie|is\s+a\s+lie|was\s+a\s+lie|said\s+no\s+one\s+ever|psych\b|sike\b)\b",
            text_lower
        ))

        # Split text into sentence/clause chunks to scope polarity to aspect context
        clauses = re.split(
            r"[.,;!?\n।|]|(?:\b(?:but|however|although|though|yet|while|on\s+the\s+other\s+hand)\b|(?<![\w\u0900-\u097F])(?:पण|परंतु|तरी|किंतु)(?![\w\u0900-\u097F]))",
            text_lower
        )

        for aspect, patterns in self.aspect_patterns.items():
            if not any(k in text_lower for k in self.aspect_keyword_sets[aspect]):
                continue

            aspect_clauses = []
            for clause in clauses:
                if any(p.search(clause) for p in patterns):
                    aspect_clauses.append(clause)

            if not aspect_clauses:
                continue

            # Determine sentiment in clauses mentioning this aspect
            pos_score = 0
            neg_score = 0

            for clause in aspect_clauses:
                words = set(re.findall(r"[\u0900-\u097F]+|\b\w+(?:-\w+)?\b", clause))
                
                # Check direct phrase matches
                for ps in POSITIVE_SIGNALS:
                    if " " in ps and ps in clause:
                        pos_score += 2
                    elif ps in words:
                        pos_score += 1

                for ns in NEGATIVE_SIGNALS:
                    if " " in ns and ns in clause:
                        neg_score += 2
                    elif ns in words:
                        neg_score += 1

                # Negation flip inside clause (English + Indic)
                if re.search(r"\b(not|never|no|hardly|scarcely)\s+(good|clean|fast|smooth|worth|impressive|responsive)\b", clause) or \
                   re.search(r"(?:चांगला|चांगली|चांगले|छान|मस्त|अच्छा|बढ़िया)\s+(?:नाही|नाहीत|नहीं)|(?:नाही|नहीं)\s+(?:चांगला|चांगली|अच्छा|छान)", clause):
                    neg_score += 2
                if re.search(r"\b(not|never|no)\s+(bad|broken|slow|crash|glitch|terrible|disaster)\b|(?:वाईट|खराब)\s+(?:नाही|नहीं)", clause):
                    pos_score += 1
                if re.search(r"\b(fixed|resolved|eliminated|no|zero|gayab)\s+.*(bug|bugs|glitch|glitches|crash|crashes|lag|issue|issues|leak|leaks)\b|\b(bug|bugs|glitch|glitches|crash|crashes|lag|issue|issues|leak|leaks)\s+(gayab|fixed|resolved|eliminated|gone)\b", clause) or \
                   re.search(r"(?:बग|बग्स|समस्या)\s+(?:नाहीत|नाही|मिटले|दूर\s+झाले)", clause):
                    pos_score += 3
                    neg_score = max(0, neg_score - 3)

            if is_sarcastic_reversal:
                # Invert polarity if sarcastic reversal was detected
                if pos_score > neg_score:
                    extracted[aspect] = "negative"
                elif neg_score > pos_score:
                    extracted[aspect] = "positive"
                else:
                    extracted[aspect] = "negative"
            else:
                if pos_score > neg_score:
                    extracted[aspect] = "positive"
                elif neg_score > pos_score:
                    extracted[aspect] = "negative"
                else:
                    extracted[aspect] = "neutral"

        return extracted
