"""Step 6: learn a "news / factual reporting" detector from our own posts.

Positives are posts whose source is an RSS news feed; negatives are posts from social platforms
(Instagram, Reddit, Twitter/X, Facebook, LinkedIn). YouTube is mixed, so it's left out. Gold-sample posts
are excluded. Both the headline alone and headline + description are used, so short titles are recognised.

Training uses scikit-learn; serving does not (src/models/news_detector.py reproduces the features).
Writes models/news_detector.npz (float16 weights, ~1 MB).

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/train_news_detector.py
"""
import json
import os
import random
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.models.news_detector import N_BUCKETS, NewsDetector, featurize  # noqa: E402
from src.preprocessing.post_text import post_to_text  # noqa: E402

SOCIAL = {"instagram", "reddit", "twitter", "x", "facebook", "linkedin"}
OUT = os.path.join(project_root, "models", "news_detector.npz")


def to_matrix(texts):
    from scipy.sparse import csr_matrix
    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        for c, v in featurize(t).items():
            rows.append(r)
            cols.append(c)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), 2 * N_BUCKETS), dtype=np.float32)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    with open(os.path.join(data_root, "data", "gold", "gold_ids.json")) as f:
        gold = set(json.load(f))
    with open(os.path.join(data_root, "data", "raw", "posts_300k.json"), encoding="utf-8") as f:
        posts = json.load(f)

    news, social, seen = [], [], set()
    for p in posts:
        if p["id"] in gold:
            continue
        src = (p.get("source") or "").lower()
        if src != "rss" and src not in SOCIAL:
            continue
        for t in {post_to_text(p.get("title"), None), post_to_text(p.get("title"), p.get("description"))}:
            if len(t) < 15 or t in seen:
                continue
            seen.add(t)
            (news if src == "rss" else social).append(t)
    random.seed(0)
    random.shuffle(news)
    random.shuffle(social)
    n = min(len(news), len(social), 60000)
    X = news[:n] + social[:n]
    y = np.array([1] * n + [0] * n)
    idx = np.random.RandomState(0).permutation(len(X))
    cut = int(len(idx) * 0.9)
    tr, te = idx[:cut], idx[cut:]
    print(f"news {len(news):,} social {len(social):,}; using {n:,} of each", flush=True)

    Xtr = to_matrix([X[i] for i in tr])
    clf = LogisticRegression(C=2.0, max_iter=2000)
    clf.fit(Xtr, y[tr])
    w = clf.coef_[0].astype(np.float16)
    np.savez_compressed(OUT, w=w, b=np.array(clf.intercept_[0], dtype=np.float32))

    det = NewsDetector(OUT)  # evaluate exactly what serving will run (float16 weights, pure numpy)
    pred = (np.array(det.proba([X[i] for i in te])) >= 0.5).astype(int)
    print(classification_report(y[te], pred, target_names=["social", "news"], digits=3))
    print(f"saved -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
