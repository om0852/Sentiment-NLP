"""Active learning: export the most useful posts for a human to label.

The teacher model and the stored DB label disagree on ~26k of the 60k labelled posts. Those are left out of
training today, but they are exactly the hard cases. A human label on them teaches the model the most.

Two kinds of rows are picked, spread across sources:
  * confident disagreements: the teacher is sure (>= 0.8) yet the DB says otherwise, so one of them is clearly wrong
  * borderline posts: the teacher's top two classes are within 0.1, so the model is unsure

Writes data/gold/review_queue.csv (gitignored, it holds user posts). Gold-sample posts are never included.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/export_review_queue.py --n 600
"""
import argparse
import collections
import csv
import json
import os
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
LABELS = ["negative", "neutral", "positive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(os.path.join(data_root, "data", "gold", "gold_ids.json")) as f:
        gold = set(json.load(f))
    confident, borderline = collections.defaultdict(list), collections.defaultdict(list)
    with open(os.path.join(data_root, "data", "silver", "teacher_labels.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in gold or r["original"] not in LABELS:
                continue
            p = sorted(r["probs"], reverse=True)
            if r["teacher"] != r["original"] and p[0] >= 0.8:
                confident[r["source"]].append(r)
            elif p[0] - p[1] < 0.1:
                borderline[r["source"]].append(r)

    rng = random.Random(args.seed)
    picked = []
    for pool, share in ((confident, 0.6), (borderline, 0.4)):
        want = int(args.n * share)
        sources = sorted(pool, key=lambda s: -len(pool[s]))
        for rows in pool.values():
            rng.shuffle(rows)
        # round-robin across sources so Instagram doesn't crowd out everything else
        i = 0
        while want > 0 and any(pool[s] for s in sources):
            s = sources[i % len(sources)]
            if pool[s]:
                picked.append((("confident_disagreement" if pool is confident else "borderline"), pool[s].pop()))
                want -= 1
            i += 1

    out = os.path.join(data_root, "data", "gold", "review_queue.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "why", "text", "db_label", "teacher_label", "teacher_confidence", "human_label"])
        for why, r in picked:
            w.writerow([r["id"], r["source"], why, r["text"], r["original"], r["teacher"], max(r["probs"]), ""])
    counts = collections.Counter((why, r["source"]) for why, r in picked)
    print(f"wrote {len(picked)} rows -> {out}")
    for (why, src), c in sorted(counts.items()):
        print(f"  {why:24} {src:10} {c}")


if __name__ == "__main__":
    main()
