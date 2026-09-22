import json
import os
import re
from typing import Dict, List, Set

class NegationHandler:
    def __init__(self, dict_path: str = None):
        if dict_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dict_path = os.path.join(base_dir, "data", "dictionaries", "negation_dict.json")
            if not os.path.exists(dict_path):
                dict_path = r"C:\Users\salun\.gemini\antigravity-ide\brain\8a79ec98-d44f-4a02-ae10-56c0e06f4c25\scratch\negation_dict.json"

        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.negation_words: Set[str] = set(data.get("negation_words", []))
        self.contractions: Dict[str, str] = data.get("contractions", {})
        
        # Punctuation or clauses that break negation scope
        self.scope_breakers = {".", "!", "?", ";", ",", "but", "however", "although", "except"}

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions like can't -> cannot."""
        if not text:
            return ""
        result = text
        for contr, expanded in self.contractions.items():
            pattern = re.compile(rf"\b{re.escape(contr)}\b", re.IGNORECASE)
            result = pattern.sub(expanded, result)
        return result

    def apply_negation_scope(self, text: str, max_window: int = 3) -> str:
        """
        Attaches a 'not_' prefix to up to `max_window` words following a negation particle
        until a scope breaker is encountered.
        e.g. 'not very good at all' -> 'not not_very not_good not_at all'
        """
        if not text:
            return ""

        # First expand contractions
        text = self.expand_contractions(text)

        tokens = re.findall(r"\w+|[^\w\s]", text)
        result_tokens = []
        
        in_negation = False
        words_remaining = 0

        for token in tokens:
            lower = token.lower()

            if lower in self.scope_breakers:
                in_negation = False
                words_remaining = 0
                result_tokens.append(token)
                continue

            if lower in self.negation_words:
                in_negation = True
                words_remaining = max_window
                result_tokens.append(token)
                continue

            if in_negation and token.isalnum():
                result_tokens.append(f"not_{token}")
                words_remaining -= 1
                if words_remaining <= 0:
                    in_negation = False
            else:
                result_tokens.append(token)

        # Reassemble
        output = " ".join(result_tokens)
        output = re.sub(r"\s+([^\w\s])", r"\1", output)  # re-attach punctuation
        return output
