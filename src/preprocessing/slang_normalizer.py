import json
import os
import re
from typing import Dict, Any, Tuple

class SlangNormalizer:
    def __init__(self, slang_dict_path: str = None):
        if slang_dict_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            slang_dict_path = os.path.join(project_root, "data", "dictionaries", "slang_dict.json")
            if not os.path.exists(slang_dict_path):
                slang_dict_path = os.path.join(os.getcwd(), "data", "dictionaries", "slang_dict.json")

        with open(slang_dict_path, "r", encoding="utf-8") as f:
            self.slang_dict: Dict[str, Dict[str, Any]] = json.load(f)

        # Sort slang items: multi-word phrases first (longest to shortest)
        multi_word = []
        single_word = []

        for term, data in self.slang_dict.items():
            if " " in term:
                multi_word.append((term, data))
            else:
                single_word.append((term, data))

        multi_word.sort(key=lambda x: len(x[0]), reverse=True)

        # Precompile patterns once in __init__
        self.multi_word_patterns = [
            (term, data, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
            for term, data in multi_word
        ]

        self.single_patterns = [
            (term, data, re.compile(rf"\b{re.escape(term)}\b(?!-)", re.IGNORECASE))
            for term, data in single_word
        ]
        self.single_word_set = {term.lower() for term, _ in single_word}

    def normalize(self, text: str, replace_with_meaning: bool = True) -> Tuple[str, float]:
        """
        Normalizes slang words in text to standard sentiment expressions.
        Returns: (normalized_text, total_slang_polarity)
        """
        if not text:
            return "", 0.0

        result = text
        text_lower = text.lower()
        total_polarity = 0.0

        # 1. Multi-word phrases first (skip regex if phrase not in text_lower)
        for phrase, data, pattern in self.multi_word_patterns:
            if phrase in text_lower:
                matches = len(pattern.findall(result))
                if matches > 0:
                    total_polarity += data.get("weight", 0.0) * matches
                    replacement = f" {data['replacement']} " if replace_with_meaning else f" _{phrase.replace(' ', '_')}_ "
                    result = pattern.sub(replacement, result)

        # 2. Single word slang with boundary (fast token set intersection filter)
        tokens = set(re.findall(r"\b\w+\b", text_lower))
        active_slang = tokens & self.single_word_set
        if active_slang:
            for term, data, pattern in self.single_patterns:
                if term.lower() in active_slang:
                    matches = len(pattern.findall(result))
                    if matches > 0:
                        total_polarity += data.get("weight", 0.0) * matches
                        replacement = f" {data['replacement']} " if replace_with_meaning else f" _{term}_ "
                        result = pattern.sub(replacement, result)

        return result, total_polarity
