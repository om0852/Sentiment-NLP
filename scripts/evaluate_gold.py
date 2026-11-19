"""Step 1: score the engine against the human-labelled gold set (see docs/LABELLING_GUIDE.md).

Reports:
  * how many rows are labelled, and how often the two annotators agree (raw % and Cohen's kappa).
    Agreement is the realistic ceiling for any model's accuracy.
  * engine accuracy on rows where the annotators agree, overall and per source,
    plus accuracy on the posts the engine does not flag as uncertain.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/evaluate_gold.py
"""
import collections
import csv
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

LABELS = ["negative", "neutral", "positive"]


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    path = os.path.join(data_root, "data", "gold", "gold_sample.csv")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    both = [r for r in rows
            if r["label_annotator_1"].strip().lower() in LABELS and r["label_annotator_2"].strip().lower() in LABELS]
    print(f"gold rows {len(rows):,}; labelled by both annotators {len(both):,}")
    if not both:
        print("Nothing to score yet. Fill in label_annotator_1 and label_annotator_2 (see docs/LABELLING_GUIDE.md).")
        return
    a = [r["label_annotator_1"].strip().lower() for r in both]
    b = [r["label_annotator_2"].strip().lower() for r in both]
    agree = [r for r, x, y in zip(both, a, b) if x == y]
    print(f"annotator agreement {len(agree) / len(both):.1%} (Cohen's kappa {cohen_kappa(a, b):.3f}) "
          f"<- realistic accuracy ceiling")

    from evaluate_cascade import Engine, score
    from src.preprocessing.context_analyzer import ContextAnalyzer
    from src.serving.cascade import load_config

    engine = Engine()
    cfg = load_config()
    items = [{"text": r["text"], "ok": {r["label_annotator_1"].strip().lower()}, "cat": r["source"]} for r in agree]
    res = engine.run(items, cfg, ContextAnalyzer(disabled_rules=cfg.get("disabled_rules", [])))
    s = score(items, res)
    print(f"engine accuracy on agreed rows {s['acc']:.1%} (n={s['n']}); transformer used on {s['escalated']:.0%}; "
          f"answered {s['coverage']:.0%} of posts with {s['answered_acc']:.1%} accuracy")
    by = collections.defaultdict(list)
    for x, r in zip(items, res):
        by[x["cat"]].append((x, r))
    for cat, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        sc = score([p for p, _ in v], [q for _, q in v])
        print(f"  {cat:12} n={sc['n']:4}  acc {sc['acc']:.1%}")


if __name__ == "__main__":
    main()
