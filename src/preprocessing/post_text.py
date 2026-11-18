"""Turn a raw scraped post (title + description) into one clean string for the transformer tier.

Scraped posts carry search-engine highlighting (<b>...</b>), HTML entities, platform suffixes
("... - Instagram") and boilerplate ("Photo by X on May 14, 2025."). None of that carries
sentiment, and it differs by source, so models would otherwise learn the source instead of the tone.
"""
import html
import re
from typing import Optional

_TAG = re.compile(r"<[^>]+>")
_PLATFORM_SUFFIX = re.compile(
    r"\s*[-|·]\s*(instagram|youtube|linkedin|facebook|twitter|x|reddit|tiktok)\s*$", re.IGNORECASE
)
_PHOTO_BOILERPLATE = re.compile(r"\bPhoto by [^.]{0,120}? on [A-Z][a-z]+ \d{1,2}, \d{4}\.?")
_VIEW_PROFILE = re.compile(r"\b(More Relevant Posts|View profile for [^.]{0,80}\.)", re.IGNORECASE)
_LEADING_ELLIPSIS = re.compile(r"^(\.\.\.|…)\s*")
_SPACE = re.compile(r"\s+")
_DOT_RUN = re.compile(r"(\s*\.\s*){3,}")


def _clean_part(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(_TAG.sub("", s))
    s = _PHOTO_BOILERPLATE.sub(" ", s)
    s = _VIEW_PROFILE.sub(" ", s)
    s = _DOT_RUN.sub(" ... ", s)
    s = _SPACE.sub(" ", s).strip()
    s = _PLATFORM_SUFFIX.sub("", s)
    s = _LEADING_ELLIPSIS.sub("", s)
    return s.strip(" |·-")


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def post_to_text(title: Optional[str] = None, description: Optional[str] = None, max_chars: int = 600) -> str:
    """Join title and description, dropping the description when it only repeats the title."""
    t, d = _clean_part(title), _clean_part(description)
    nt, nd = _norm(t), _norm(d)
    if not nd or (nt and (nd in nt or nt in nd) and len(nd) <= len(nt)):
        text = t
    elif not nt or nd.startswith(nt[: max(10, len(nt) - 5)]):
        text = d
    else:
        text = f"{t}. {d}" if not t.endswith((".", "!", "?", "…")) else f"{t} {d}"
    return text[:max_chars].strip()
