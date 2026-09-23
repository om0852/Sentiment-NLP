import json
import os
import re
from typing import Dict, Any

class EmojiMapper:
    def __init__(self, emoji_dict_path: str = None):
        if emoji_dict_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            emoji_dict_path = os.path.join(project_root, "data", "dictionaries", "emoji_dict.json")
            if not os.path.exists(emoji_dict_path):
                emoji_dict_path = os.path.join(os.getcwd(), "data", "dictionaries", "emoji_dict.json")

        with open(emoji_dict_path, "r", encoding="utf-8") as f:
            self.emoji_dict: Dict[str, Dict[str, Any]] = json.load(f)

        sorted_emojis = sorted(self.emoji_dict.keys(), key=len, reverse=True)
        self.emoji_pattern = re.compile("|".join(re.escape(e) for e in sorted_emojis))

    def map_emojis(self, text: str, mode: str = "token") -> str:
        """
        Replaces emojis in text.
        mode="token": replaces with '_emoji_fire_', '_emoji_skull_'
        mode="meaning": replaces with descriptive words e.g. 'fire awesome amazing'
        """
        if not text or text.isascii():
            return text

        def _replace(match):
            e = match.group(0)
            data = self.emoji_dict.get(e)
            if not data:
                return e
            return f" {data['token']} " if mode == "token" else f" {data['meaning']} "

        return self.emoji_pattern.sub(_replace, text)

    def get_emoji_polarity(self, text: str) -> float:
        """Returns total polarity score from emojis in text."""
        if not text or text.isascii():
            return 0.0

        score = 0.0
        for match in self.emoji_pattern.finditer(text):
            data = self.emoji_dict.get(match.group(0))
            if data:
                score += data.get("weight", 0.0)
        return score
