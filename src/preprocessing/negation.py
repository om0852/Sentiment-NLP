import json
import os
import re
from typing import Dict, List, Set

class NegationHandler:
    def __init__(self, dict_path: str = None):
        if dict_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            dict_path = os.path.join(project_root, "data", "dictionaries", "negation_dict.json")
            if not os.path.exists(dict_path):
                dict_path = os.path.join(os.getcwd(), "data", "dictionaries", "negation_dict.json")

        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.negation_words: Set[str] = set(data.get("negation_words", []))
        self.contractions: Dict[str, str] = data.get("contractions", {})
        
        # Punctuation or clauses that break negation scope
        self.scope_breakers = {".", "!", "?", ";", ",", "but", "however", "although", "except"}

        # Precompile combined contraction regex
        sorted_contr = sorted(self.contractions.keys(), key=len, reverse=True)
        self.contr_pattern = re.compile(rf"\b({'|'.join(re.escape(c) for c in sorted_contr)})\b", re.IGNORECASE)
        self.contr_map = {c.lower(): exp for c, exp in self.contractions.items()}

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions like can't -> cannot."""
        if not text or ("'" not in text and "’" not in text):
            return text
        return self.contr_pattern.sub(lambda m: self.contr_map.get(m.group(0).lower(), m.group(0)), text)

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
        if not any(t.lower() in self.negation_words for t in tokens):
            return text

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
