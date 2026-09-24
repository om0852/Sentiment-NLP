import re
from typing import Dict, Any, Tuple
from .cleaner import TextCleaner
from .emoji_mapper import EmojiMapper
from .slang_normalizer import SlangNormalizer
from .negation import NegationHandler
from .indic_normalizer import IndicNormalizer

class PreprocessingPipeline:
    def __init__(self, slang_dict_path: str = None, emoji_dict_path: str = None, 
                 negation_dict_path: str = None, indic_dict_path: str = None):
        self.cleaner = TextCleaner()
        self.indic_normalizer = IndicNormalizer(indic_dict_path)
        self.emoji_mapper = EmojiMapper(emoji_dict_path)
        self.slang_normalizer = SlangNormalizer(slang_dict_path)
        self.negation_handler = NegationHandler(negation_dict_path)
        
        # Hashtag splitter pattern (splits CamelCase like #NeverAgain -> Never Again)
        self.hashtag_pattern = re.compile(r"#([a-zA-Z0-9_]+)")

    def split_hashtags(self, text: str) -> str:
        """Splits hashtags into separate words while preserving semantic meaning."""
        if "#" not in text:
            return text

        def _split_tag(match):
            tag = match.group(1)
            # split PascalCase / camelCase
            words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", tag)
            if words:
                return " " + " ".join(words) + " "
            return " " + tag + " "
            
        return self.hashtag_pattern.sub(_split_tag, text)

    def process(self, text: str, for_training: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Runs the full preprocessing pipeline:
        1. Clean HTML, URLs, repeated characters, detect sarcasm markers
        2. Indic Devanagari normalization & negation scoping (Hindi & Marathi)
        3. Split hashtags
        4. Map emojis to sentiment tokens & extract emoji polarity
        5. Normalize slang terms to standard semantic equivalents (including Hinglish & Maranglish)
        6. Apply English negation scoping (attaching not_ prefix)
        Returns: (processed_text, metadata_dict)
        """
        if not text or not isinstance(text, str):
            return "", {"sarcasm_detected": False, "emoji_polarity": 0.0, "slang_polarity": 0.0, "indic_polarity": 0.0}

        # 1. Clean & detect sarcasm
        cleaned, has_sarcasm = self.cleaner.clean(text)

        # 2. Indic Devanagari normalization (fast early return for non-Devanagari)
        indic_processed, indic_polarity, is_indic = self.indic_normalizer.normalize(cleaned)

        # 3. Split hashtags (fast check)
        hashtag_expanded = self.split_hashtags(indic_processed)

        # 4. Emoji mapping (fast ascii check)
        emoji_polarity = self.emoji_mapper.get_emoji_polarity(hashtag_expanded)
        emoji_processed = self.emoji_mapper.map_emojis(hashtag_expanded, mode="meaning")

        # 5. Slang normalization (includes Hinglish & Maranglish)
        slang_processed, slang_polarity = self.slang_normalizer.normalize(emoji_processed, replace_with_meaning=True)

        # 6. Negation handling
        final_text = self.negation_handler.apply_negation_scope(slang_processed)
        final_text = re.sub(r"\s+", " ", final_text).strip().lower()

        metadata = {
            "sarcasm_detected": has_sarcasm,
            "emoji_polarity": emoji_polarity,
            "slang_polarity": slang_polarity,
            "indic_polarity": indic_polarity,
            "is_indic": is_indic,
            "composite_polarity": emoji_polarity + slang_polarity + indic_polarity
        }

        return final_text, metadata
