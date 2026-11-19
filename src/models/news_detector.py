"""News / factual-reporting detector (step 6) as a tiny hashed linear model.

Pure Python + NumPy at serving time (no scikit-learn, which alone costs ~80 MB of RAM).
Features: word 1-2 grams (case kept) and character 3-4 grams inside words (first 300 characters) (lower-cased), each block hashed
into 2^18 buckets with CRC32 and L2-normalised. Weights come from scripts/train_news_detector.py.
"""
import os
import re
import zlib
from typing import Dict, List

import numpy as np

N_BUCKETS = 2 ** 18
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) % N_BUCKETS


def _block(keys: List[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k in keys:
        i = _h(k)
        out[i] = out.get(i, 0.0) + 1.0
    norm = sum(v * v for v in out.values()) ** 0.5 or 1.0
    return {i: v / norm for i, v in out.items()}


def featurize(text: str, max_chars: int = 300) -> Dict[int, float]:
    """Sparse feature vector: indices [0, N) word block, [N, 2N) char block."""
    text = text[:max_chars]
    toks = _TOKEN.findall(text)
    words = [f"w:{t}" for t in toks] + [f"b:{a} {b}" for a, b in zip(toks, toks[1:])]
    chars = []
    for w in text.lower().split():
        w = f" {w} "
        for n in (3, 4):
            chars.extend(w[i: i + n] for i in range(len(w) - n + 1))
    feats = _block(words)
    for i, v in _block(chars).items():
        feats[N_BUCKETS + i] = v
    return feats


class NewsDetector:
    def __init__(self, path: str):
        data = np.load(path)
        self.w = data["w"].astype(np.float32)
        self.b = float(data["b"])

    @staticmethod
    def available(path: str) -> bool:
        return os.path.exists(path)

    def proba(self, texts: List[str]) -> List[float]:
        out = []
        for t in texts:
            f = featurize(t)
            idx = np.fromiter(f.keys(), dtype=np.int64, count=len(f))
            val = np.fromiter(f.values(), dtype=np.float32, count=len(f))
            z = self.b + float(self.w[idx] @ val)
            out.append(1.0 / (1.0 + float(np.exp(-z))))
        return out
