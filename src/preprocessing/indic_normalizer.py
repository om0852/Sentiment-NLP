import os
import re
import json
from typing import Dict, Any, Tuple

class IndicNormalizer:
    """
    Lightweight, high-speed normalizer for Hindi and Marathi in Devanagari script.
    Zero external dependencies, sub-millisecond execution.
    - Detects Devanagari Unicode range [\u0900-\u097F]
    - Maps Devanagari sentiment roots and multi-word idioms
    - Applies native Indic negation scoping (e.g., 'चांगला नाही' -> negative, 'बग्स नाहीत' -> positive)
    - Projects Devanagari sentiment to normalized semantic tokens for high-precision TF-IDF classification
    """
    def __init__(self, dict_path: str = None):
        if dict_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            dict_path = os.path.join(project_root, "data", "dictionaries", "indic_dict.json")
            if not os.path.exists(dict_path):
                dict_path = os.path.join(os.getcwd(), "data", "dictionaries", "indic_dict.json")

        self.devanagari_pattern = re.compile(r"[\u0900-\u097F]")
        self.phrases = {}
        self.positive_words = {}
        self.negative_words = {}
        self.negation_words = set()

        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phrases = data.get("phrases", {})
                self.positive_words = data.get("positive_words", {})
                self.negative_words = data.get("negative_words", {})
                self.negation_words = set(data.get("negation_words", []))

        # Precompile multi-word phrase regexes sorted by length descending
        sorted_phrases = sorted(self.phrases.keys(), key=len, reverse=True)
        self.phrase_patterns = [
            (phrase, self.phrases[phrase], re.compile(rf"{re.escape(phrase)}", re.IGNORECASE))
            for phrase in sorted_phrases
        ]

        # Resolution words (e.g., 'मिटले', 'झाले', 'दूर')
        self.resolution_words = {"मिटले", "मिटला", "दूर", "सॉल्व्ह", "दुरुस्त", "झाले", "झाला"}

    def is_indic(self, text: str) -> bool:
        """Returns True if text contains Devanagari script."""
        return bool(self.devanagari_pattern.search(text))

    def normalize(self, text: str) -> Tuple[str, float, bool]:
        """
        Normalizes Devanagari Hindi and Marathi text.
        Returns: (normalized_text, total_polarity, is_devanagari)
        """
        if not text or not self.is_indic(text):
            return text, 0.0, False

        result = text
        total_polarity = 0.0

        # 1. Multi-word phrase matching
        for phrase, data, pat in self.phrase_patterns:
            if phrase in result:
                matches = len(pat.findall(result))
                if matches > 0:
                    total_polarity += data.get("weight", 0.0) * matches
                    replacement = f" {data['replacement']} "
                    result = pat.sub(replacement, result)

        # 2. Tokenize and handle adjacent negation and resolution
        tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z0-9]+|[^\w\s]", result)
        output_tokens = []
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]

            # Check next token for post-negation (e.g. 'चांगला नाही', 'अच्छा नहीं')
            next_tok = tokens[i + 1] if i + 1 < n else ""
            prev_tok = tokens[i - 1] if i > 0 else ""

            # Check if current token is negated by previous or next token
            is_negated = (next_tok in self.negation_words) or (prev_tok in self.negation_words)

            if tok in self.positive_words:
                pw_data = self.positive_words[tok]
                if is_negated:
                    # 'चांगला नाही' -> flip to negative
                    total_polarity -= 2.0
                    output_tokens.append(f"not_{pw_data['replacement']} terrible bad")
                    if next_tok in self.negation_words:
                        i += 1  # Skip the negation token
                else:
                    total_polarity += pw_data.get("weight", 1.5)
                    output_tokens.append(pw_data["replacement"])

            elif tok in self.negative_words:
                nw_data = self.negative_words[tok]
                # Check for negation or resolution (e.g. 'बग्स नाहीत', 'बग्स मिटले')
                is_resolved = is_negated or (next_tok in self.resolution_words) or (prev_tok in self.resolution_words)
                if is_resolved:
                    total_polarity += 1.8
                    output_tokens.append("fixed resolved clean no bugs")
                    if next_tok in self.negation_words or next_tok in self.resolution_words:
                        i += 1
                else:
                    total_polarity += nw_data.get("weight", -1.5)
                    output_tokens.append(nw_data["replacement"])

            elif tok in self.negation_words:
                # Standalone negation word
                output_tokens.append("not")
            else:
                output_tokens.append(tok)

            i += 1

        normalized_text = " ".join(output_tokens)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        return normalized_text, total_polarity, True
