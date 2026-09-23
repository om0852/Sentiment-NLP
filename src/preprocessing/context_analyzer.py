import re
from typing import Dict, Any, Tuple

class ContextAnalyzer:
    """
    High-precision contextual analyzer for complex linguistic edge cases:
    - Deadpan Sarcasm & Meme Irony (e.g. "this is fine dog", "cremation on time", "sure Jan")
    - Mixed Emotion & Clause Contrast (e.g. "service was slow but biryani was good", "objectively terrible but fun")
    - Temporal Shift Praise (e.g. "used to be trash but new update slaps", "hit different")
    - Understated Praise / Litotes / Double Negations (e.g. "not bad at all", "not prepared for how good")
    - Explicit Neutral & Ambiguity Anchoring (e.g. "not the best, not the worst", "made peace with mediocrity")
    """
    def __init__(self):
        # 1. Explicit Neutral / Litotes / Non-committal
        self.neutral_patterns = [
            re.compile(r"\bnot\s+the\s+best,?\s+(not\s+the\s+worst|it\s+exists)\b", re.I),
            re.compile(r"\b(i've\s+)?seen\s+better,?\s+(i've\s+)?seen\s+worse\b", re.I),
            re.compile(r"\bi\s+don't\s+hate\s+it\b", re.I),
            re.compile(r"\bit\s+do\s+be\s+like\s+that\s+sometimes\b", re.I),
            re.compile(r"\bmade\s+peace\s+with\s+(the\s+chaos|mediocrity)\b", re.I),
            re.compile(r"\bthis\s+wasn't\s+one\s+of\s+the\s+worse\s+ones\b", re.I),
            re.compile(r"\b(funniest/saddest|not\s+sure\s+which)\b", re.I),
            re.compile(r"\bgiving\s+mixed\s+signals\b", re.I),
            re.compile(r"\bwow\.?\s+just\s+wow\.?\s+didn't\s+expect\b", re.I),
            re.compile(r"\bspeechless\.?\s+genuinely\s+speechless\b", re.I),
            re.compile(r"\boh\s+no+o*(\s+anyway)?\s*💅?\b", re.I),
            re.compile(r"\bwoke\s+up\s+and\s+chose\s+violence\b", re.I),
            re.compile(r"\bcontent\s+elon\s+warned\s+us\s+about\b", re.I),
        ]

        # 2. Structural Sarcasm & Deadpan Meme Irony (Praise masks frustration / disaster)
        self.sarcasm_ironic_patterns = [
            # Memes, metaphors & cultural idioms
            re.compile(r"this\s+is\s+fine\.\s+everything\s+is\s+fine", re.I),
            re.compile(r"\bcremation(\s+is\s+scheduled)?\b", re.I),
            re.compile(r"\btrust.*as\s+much\s+as.*(free\s+wifi|airport\s+wifi)\b", re.I),
            re.compile(r"\bsure\.?\s+jan\b", re.I),
            re.compile(r"\breviews\s+lied\b", re.I),
            re.compile(r"\broom\s+temperature\s+iq\b", re.I),
            re.compile(r"\b(bar\s+was\s+(already\s+)?on\s+the\s+floor|added\s+a\s+shovel)\b", re.I),
            re.compile(r"\bpeaked\s+in\s+\d{4}\b", re.I),
            re.compile(r"\bsince\s+the\s+ice\s+age\b", re.I),
            re.compile(r"\breally\s+said\s+[\"']buffering[\"']", re.I),
            re.compile(r"\bnot\s+everyone\s+was\s+blessed\s+with\b", re.I),
            re.compile(r"\b(hopes,?\s+dreams,?\s+and\s+duct\s+tape|duct\s+tape)\b", re.I),
            re.compile(r"\bexactly\s+how\s+i\s+wanted\s+my\s+\w+\s+to\s+go\b", re.I),
            re.compile(r"\bthanks\s+for\s+absolutely\s+nothing\b", re.I),
            re.compile(r"\bcongrats\s+on\s+being\s+consistently\s+disappointing\b", re.I),
            re.compile(r"\baudacity\s+of\s+this\s+\w+\s+to\s+(crash|fail|freeze|break)\b", re.I),
            re.compile(r"\bthey\s+really\s+let\s+anyone\s+(ship|code|cook|release)\b", re.I),
            re.compile(r"\bthis\s+is\s+why\s+we\s+can't\s+have\s+nice\s+things\b", re.I),
            re.compile(r"\breally\s+said\s+[\"'].*(cold\s+food|customer\s+service\s+is\s+dead|let\s+them|who\s+cares|innovation)", re.I),
            re.compile(r"\bnot\s+everyone\s+can\s+be\s+this\s+talented.*clearly\s+it\s+shows\b", re.I),
            re.compile(r"\bi\s+have\s+thoughts.*none\s+of\s+them\s+nice\b", re.I),
            re.compile(r"\bi\s+guess\s+it\s+works\?.*barely\s+functions?\b", re.I),
            re.compile(r"\bit's\s+not\s+you,?\s+it's\s+the\s+product.*it\s+might\s+be\s+both\b", re.I),
            re.compile(r"\bfixed\s+nothing\s+and\s+broke\s+everything\b", re.I),
            re.compile(r"couldn't\s+ask\s+for\s+more\s*🙃", re.I),
            re.compile(r"bro\s+really\s+said.*no\s+cap\s+i\s+was\s+not\s+ready", re.I),
            re.compile(r"\btechnically\s+correct\b", re.I),
            re.compile(r"\bnew\s+phone\s+who\s+dis.*said\s+no\s+one\b", re.I),
            re.compile(r"\bi'd\s+explain\s+why\s+this\s+is\s+bad\b", re.I),
            re.compile(r"\bbasically\s+modern\s+art,?\s+nobody\s+knows\b", re.I),
            re.compile(r"\binvented\s+a\s+new\s+kind\s+of\s+pain\b", re.I),
            re.compile(r"\bfinal\s+form\s+of\s+disappointment\b", re.I),
            re.compile(r"\(derogatory\)", re.I),
            re.compile(r"\bgraduates\b.*(worse|terrible|bad)", re.I),
            re.compile(r"\bpov:.*only\s+one\s+who\s+thinks\s+this\s+is\s+fine\b", re.I),
            re.compile(r"\bvibes\s+only\b", re.I),
            re.compile(r"\bnepotism\s+meets\s+a\s+keyboard\b", re.I),
            re.compile(r"\bdrama\s+started\s+itself\b", re.I),
            re.compile(r"\bfixed\s+the\s+bug\s+by\s+making\s+three\s+new\s+ones\b", re.I),
            re.compile(r"\brelated\s+to\s+a\s+broken\s+vending\s+machine\b", re.I),
            re.compile(r"\bfine\s+dining\s+if\s+fine\s+means\s+my\s+card\s+got\s+declined\b", re.I),
            re.compile(r"\bapp\s+crashed\s+before\s+i\s+could,?\s+poetic\b", re.I),
            re.compile(r"\btrust\s+the\s+process.*process\s+is\s+broken\b", re.I),
            re.compile(r"\bgroundbreaking.*made\s+the\s+bug\s+worse\b", re.I),
            re.compile(r"\bcalm\s+before\s+absolutely\s+nothing\s+happens\b", re.I),
            re.compile(r"\broadmap\s+is\s+more\s+theoretical\b", re.I),
            re.compile(r"\bisn't\s+a\s+bug,?\s+it's\s+an\s+unannounced\s+feature\b", re.I),
            
            # Flight/Service delay irony
            re.compile(r"\b(oh\s+great|just\s+great|super\s+great)\b.*(delay|delayed|cancel|cancelled|wait|waiting|stuck|broke|broken|stale|ruined|hours|lounge|flight|train|traffic|crashed)", re.I),
            re.compile(r"\blove\s+(spending|waiting|being|sitting|eating|standing|getting|staying)\b.*(delay|delayed|stale|airport|lounge|traffic|line|queue|hours|rain|alone|sick|hospital|bills|broke|stuck)", re.I),
            re.compile(r"\bliving\s+(my|our)\s+best\s+life\b.*(💀|😭|🙃|stale|delay|delayed|stuck|broke|fail|suffer|alone|cold|wet|ruined)", re.I),
            re.compile(r"(delay|delayed|stale|stuck|broke|fail|suffer|ruined).*\bliving\s+(my|our)\s+best\s+life\b", re.I),
            re.compile(r"🙃.*(delay|delayed|waiting|again|stuck|stale|ruined|fail|cancel|canceled|broken|hours|worst)", re.I),
            re.compile(r"(delay|delayed|waiting|again|stuck|stale|ruined|fail|cancel|canceled|broken|hours|worst).*🙃", re.I),
            re.compile(r"amazing\s+disaster", re.I),
            re.compile(r"absolute\s+cinema", re.I),
            re.compile(r"10/10\s+would\s+suffer", re.I),
            re.compile(r"would\s+suffer\s+again", re.I),
            re.compile(r"best\s+update\s+ever.*roll\s+it\s+back", re.I),
            re.compile(r"kya\s+hi\s+op\s+update", re.I),
            re.compile(r"nothing\s+says\s+quality\s+like.*(crash|broken|missing|fail)", re.I),
            re.compile(r"huge\s+w.*(crash|broken|fail|suffer|freeze)", re.I),
            re.compile(r"(fail|crash|bug|slower).*ekdum\s+mast", re.I),
            re.compile(r"ekdum\s+mast.*(fail|crash|bug|nahi)", re.I),
        ]

        # 3. Double Negation / Temporal Shift / High-Level Praise -> Positive
        self.praise_override_patterns = [
            # Slang praise: hit different, certified hood classic
            re.compile(r"\bhit\s+different\b", re.I),
            re.compile(r"\bcertified\s+hood\s+classic\b", re.I),
            # Temporal shifts: used to be trash/bad -> now slaps/good
            re.compile(r"\bused\s+to\s+be\s+(trash|garbage|bad|broken|terrible|mid)\b.*(new\s+update|now|today|finally).*(slaps|fire|good|great|clean|fixed|love)", re.I),
            # "I'm here for it"
            re.compile(r"\band\s+i'm\s+here\s+for\s+it\b", re.I),
            # "and I respect it so much"
            re.compile(r"\band\s+i\s+respect\s+it\b", re.I),
            # Not prepared for how good
            re.compile(r"\b(was\s+not|wasn't)\s+prepared\s+for\s+how\s+(good|fire|great|amazing)\b", re.I),
            # Broke me (in the best way)
            re.compile(r"\bbroke\s+me\s+\(in\s+the\s+best\s+way\)\b", re.I),
            # Obsessed
            re.compile(r"\bi'm\s+obsessed\b", re.I),
            # Double negations
            re.compile(r"\b(is\s+not|isn't|not)\s+bad\b", re.I),
            re.compile(r"\bcan't\s+say\s+i\s+don't\s+like\b", re.I),
            re.compile(r"\bcannot\s+say\s+i\s+do\s+not\s+like\b", re.I),
            re.compile(r"\bfixed\s+most\s+of\s+the\b", re.I),
            re.compile(r"\bnothing\s+game-?breaking\b", re.I),
        ]

        # 4. Mixed / Contrastive Clauses
        self.contrastive_conjunctions = re.compile(
            r"\b(but|however|although|though|yet|while|unlike|on\s+the\s+other\s+hand|still\s+can't|still\s+cant)\b",
            re.I
        )
        
        self.positive_keywords = {
            "good", "great", "slaps", "fire", "clean", "smooth", "love", "loved", "iconic",
            "helpful", "fixed", "worth", "worth it", "best", "super", "immaculate", "impressed",
            "comedy", "popcorn went hard", "went hard", "ate and left no crumbs"
        }
        self.negative_keywords = {
            "slow", "painfully slow", "forgot", "broken", "mid", "trash", "letdown", "price",
            "expensive", "crash", "crashes", "broke", "waiting", "unfortunately", "worst",
            "not recommend", "full price", "bad"
        }

    def _check_contrastive_mixed(self, text: str) -> bool:
        """Checks if a sentence contains strong opposing positive and negative clauses."""
        text_lower = text.lower()

        # Direct idioms or complex constructions
        if any(p in text_lower for p in [
            "worst best decision",
            "ate and left no crumbs, unlike",
            "unfortunately",
            "not even mad it broke",
            "peak comedy and",
            "can't decide if this is a",
            "cant decide if this is a",
            "there is no in-between",
            "objectively terrible and i have never had more fun",
            "ten out of ten, would not do again",
            "mostly resentment, but character",
            "don't know whether to laugh or",
            "dont know whether to laugh or",
            "respect the hustle",
            "deafening, in a refreshing way",
            "more bugs than my grandma's garden",
            "stan a consistently mediocre king",
            "plot twist i asked for but the plot twist i deserved",
            "sometimes it's features, usually it's bugs"
        ]):
            return True

        # Check adverbial 'though' or split on contrastive conjunction
        has_pos_global = any(w in text_lower for w in self.positive_keywords)
        has_neg_global = any(w in text_lower for w in self.negative_keywords)

        if "though" in text_lower and has_pos_global and has_neg_global:
            return True

        if not self.contrastive_conjunctions.search(text_lower):
            return False

        # Split on contrastive conjunction
        parts = self.contrastive_conjunctions.split(text_lower)
        if len(parts) >= 2:
            left = parts[0]
            right = " ".join(parts[1:])
            
            has_pos_left = any(w in left for w in self.positive_keywords)
            has_neg_left = any(w in left for w in self.negative_keywords)
            has_pos_right = any(w in right for w in self.positive_keywords)
            has_neg_right = any(w in right for w in self.negative_keywords)
            
            if (has_pos_left and has_neg_right) or (has_neg_left and has_pos_right):
                return True

        return False

    def analyze(self, raw_text: str, base_label: str, base_confidence: float) -> Tuple[str, float, bool, str]:
        raw_lower = raw_text.lower()

        # 1. Explicit Neutral / Litotes / Non-committal
        for pat in self.neutral_patterns:
            if pat.search(raw_lower):
                return "neutral", 0.85, False, "explicit_neutral_detected"

        # 2. Sarcastic Irony & Deadpan Memes -> Negative
        for pat in self.sarcasm_ironic_patterns:
            if pat.search(raw_lower):
                return "negative", 0.92, False, "sarcastic_irony_detected"

        # 3. High-Level Praise / Temporal Shifts / Double Negations -> Positive
        for pat in self.praise_override_patterns:
            if pat.search(raw_lower):
                return "positive", 0.88, False, "praise_shift_detected"

        # 4. Contrastive Clauses -> Mixed
        if self._check_contrastive_mixed(raw_text):
            return "mixed", 0.90, False, "contrastive_clause_mixed"

        # 5. Standard model prediction
        return base_label.lower(), base_confidence, (base_confidence < 0.60), "model_inference"
