import json
import os
from typing import Dict, Any

class EmojiMapper:
    def __init__(self, emoji_dict_path: str = None):
        if emoji_dict_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            emoji_dict_path = os.path.join(base_dir, "data", "dictionaries", "emoji_dict.json")
            if not os.path.exists(emoji_dict_path):
                # Fallback to local scratch
                emoji_dict_path = r"C:\Users\salun\.gemini\antigravity-ide\brain\8a79ec98-d44f-4a02-ae10-56c0e06f4c25\scratch\emoji_dict.json"

        with open(emoji_dict_path, "r", encoding="utf-8") as f:
            self.emoji_dict: Dict[str, Dict[str, Any]] = json.load(f)

    def map_emojis(self, text: str, mode: str = "token") -> str:
        """
        Replaces emojis in text.
        mode="token": replaces with '_emoji_fire_', '_emoji_skull_'
        mode="meaning": replaces with descriptive words e.g. 'fire awesome amazing'
        """
        if not text:
            return ""

        result = text
        for emoji, data in self.emoji_dict.items():
            if emoji in result:
                replacement = f" {data['token']} " if mode == "token" else f" {data['meaning']} "
                result = result.replace(emoji, replacement)

        return result

    def get_emoji_polarity(self, text: str) -> float:
        """Returns total polarity score from emojis in text."""
        score = 0.0
        for emoji, data in self.emoji_dict.items():
            count = text.count(emoji)
            if count > 0:
                score += data.get("weight", 0.0) * count
        return score
