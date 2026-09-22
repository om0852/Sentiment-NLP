import json
import os
import re
from typing import Dict, Any, Tuple

class SlangNormalizer:
    def __init__(self, slang_dict_path: str = None):
        if slang_dict_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            slang_dict_path = os.path.join(base_dir, "data", "dictionaries", "slang_dict.json")
            if not os.path.exists(slang_dict_path):
                slang_dict_path = r"C:\Users\salun\.gemini\antigravity-ide\brain\8a79ec98-d44f-4a02-ae10-56c0e06f4c25\scratch\slang_dict.json"

        with open(slang_dict_path, "r", encoding="utf-8") as f:
            self.slang_dict: Dict[str, Dict[str, Any]] = json.load(f)

        # Sort slang items: multi-word phrases first (longest to shortest)
        self.multi_word_slang = []
        self.single_word_slang = []

        for term, data in self.slang_dict.items():
            if " " in term:
                self.multi_word_slang.append((term, data))
            else:
                self.single_word_slang.append((term, data))

        self.multi_word_slang.sort(key=lambda x: len(x[0]), reverse=True)

        # Precompile regex patterns for single word slang
        self.single_patterns = []
        for term, data in self.single_word_slang:
            # Word boundary pattern that also rejects trailing hyphen (e.g., 'mid-michigan')
            pattern = re.compile(rf"\b{re.escape(term)}\b(?!-)", re.IGNORECASE)
            self.single_patterns.append((term, data, pattern))

    def normalize(self, text: str, replace_with_meaning: bool = True) -> Tuple[str, float]:
        """
        Normalizes slang words in text to standard sentiment expressions.
        Returns: (normalized_text, total_slang_polarity)
        """
        if not text:
            return "", 0.0

        result = text
        total_polarity = 0.0

        # 1. Multi-word phrases first
        for phrase, data in self.multi_word_slang:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            matches = len(pattern.findall(result))
            if matches > 0:
                total_polarity += data.get("weight", 0.0) * matches
                replacement = f" {data['replacement']} " if replace_with_meaning else f" _{phrase.replace(' ', '_')}_ "
                result = pattern.sub(replacement, result)

        # 2. Single word slang with boundary
        for term, data, pattern in self.single_patterns:
            matches = len(pattern.findall(result))
            if matches > 0:
                total_polarity += data.get("weight", 0.0) * matches
                replacement = f" {data['replacement']} " if replace_with_meaning else f" _{term}_ "
                result = pattern.sub(replacement, result)

        return result, total_polarity
