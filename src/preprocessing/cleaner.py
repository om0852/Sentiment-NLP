import re
import html
from typing import Tuple

class TextCleaner:
    def __init__(self):
        # Match URLs
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        # Match user mentions e.g. @username
        self.mention_pattern = re.compile(r"@[\w_]+", re.IGNORECASE)
        # Match HTML tags
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        # Match 3+ repeated characters e.g. 'sooooo' -> 'soo'
        self.repeated_char_pattern = re.compile(r"(.)\1{2,}")
        # Match multiple whitespace
        self.whitespace_pattern = re.compile(r"\s+")
        # Sarcasm markers
        self.sarcasm_markers = [
            re.compile(r"(?:^|\s)/s(?:\s|$|[.,!?])", re.IGNORECASE),
            re.compile(r"\byeah right\b", re.IGNORECASE),
            re.compile(r"\btotally normal\b", re.IGNORECASE),
            re.compile(r"\bas if\b", re.IGNORECASE),
            re.compile(r"\bgenius move\b", re.IGNORECASE),
            re.compile(r"\bbig surprise\b", re.IGNORECASE),
        ]

    def detect_sarcasm_marker(self, text: str) -> bool:
        """Returns True if explicit sarcasm markers are present."""
        for pattern in self.sarcasm_markers:
            if pattern.search(text):
                return True
        return False

    def clean(self, text: str, keep_sarcasm_tag: bool = True) -> Tuple[str, bool]:
        """
        Cleans the input text and flags if explicit sarcasm markers are present.
        Returns: (cleaned_text, has_sarcasm_marker)
        """
        if not text or not isinstance(text, str):
            return "", False

        # Unescape HTML entities (&amp;, &lt;, etc.)
        cleaned = html.unescape(text)

        # Check for sarcasm indicators before stripping
        has_sarcasm = self.detect_sarcasm_marker(cleaned)

        # Remove HTML tags
        cleaned = self.html_tag_pattern.sub(" ", cleaned)

        # Remove URLs
        cleaned = self.url_pattern.sub(" ", cleaned)

        # Normalize Twitter/IG mentions to a generic token or strip
        cleaned = self.mention_pattern.sub(" ", cleaned)

        # Collapse elongated characters (e.g. 'soooo' -> 'soo', 'loooove' -> 'loove')
        cleaned = self.repeated_char_pattern.sub(r"\1\1", cleaned)

        # Mark sarcasm tag explicitly in text for feature extraction
        if has_sarcasm:
            cleaned = re.sub(r"(?:^|\s)/s(?:\s|$|[.,!?])", " _sarcasm_tag_ ", cleaned)

        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()

        return cleaned, has_sarcasm
