"""Steps 1 + 2 of the accuracy plan.

1. Holds out a random gold sample of real posts for humans to label (data/gold/gold_sample.csv).
   These posts are never used for training, labelling or threshold tuning.
2. Labels a larger pool of real posts with a local multilingual teacher model
   (cardiffnlp/twitter-xlm-roberta-base-sentiment) and writes data/silver/teacher_labels.jsonl.
   Nothing leaves this machine. Resumable: re-running skips posts already labelled.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/build_silver_dataset.py --pool 60000
"""
import argparse
import csv
import json
import os
import random
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.preprocessing.post_text import post_to_text, _norm  # noqa: E402

TEACHER = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ["negative", "neutral", "positive"]


def load_posts():
    path = os.path.join(data_root, "data", "raw", "posts_300k.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    seen, posts = set(), []
    for p in raw:
        text = post_to_text(p.get("title"), p.get("description"))
        key = _norm(text)
        if len(key) < 5 or key in seen:
            continue
        seen.add(key)
        posts.append({
            "id": p["id"],
            "source": (p.get("source") or "").lower(),
            "text": text,
            "original": (p.get("original_sentiment") or "").strip().lower(),
        })
    return posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    posts = load_posts()
    print(f"unique cleaned posts: {len(posts):,}", flush=True)
    random.Random(args.seed).shuffle(posts)

    gold_dir = os.path.join(data_root, "data", "gold")
    os.makedirs(gold_dir, exist_ok=True)
    gold_csv = os.path.join(gold_dir, "gold_sample.csv")
    gold = posts[: args.gold]
    if not os.path.exists(gold_csv):
        with open(gold_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "text", "label_annotator_1", "label_annotator_2", "notes"])
            for p in gold:
                w.writerow([p["id"], p["source"], p["text"], "", "", ""])
        print(f"wrote {len(gold)} gold posts for human labelling -> {gold_csv}", flush=True)
    with open(os.path.join(gold_dir, "gold_ids.json"), "w") as f:
        json.dump([p["id"] for p in gold], f)

    pool = posts[args.gold: args.gold + args.pool]
    silver_dir = os.path.join(data_root, "data", "silver")
    os.makedirs(silver_dir, exist_ok=True)
    out_path = os.path.join(silver_dir, "teacher_labels.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [p for p in pool if p["id"] not in done]
    print(f"pool {len(pool):,}; already labelled {len(done):,}; to label {len(todo):,}", flush=True)
    if not todo:
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER).eval()

    # Length-sorted batches waste far less padding on CPU.
    todo.sort(key=lambda p: len(p["text"]))
    t0, n = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as out, torch.inference_mode():
        for i in range(0, len(todo), args.batch):
            batch = todo[i: i + args.batch]
            enc = tok([p["text"] for p in batch], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
            for p, pr in zip(batch, probs):
                rec = dict(p, teacher=LABELS[max(range(3), key=pr.__getitem__)],
                           probs=[round(x, 4) for x in pr])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(batch)
            if (i // args.batch) % 50 == 0:
                out.flush()
                rate = n / (time.time() - t0)
                print(f"{n:,}/{len(todo):,} labelled, {rate:.1f} posts/s, "
                      f"eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
