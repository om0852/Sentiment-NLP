"""Soft labels from an English sentiment teacher (cardiffnlp/twitter-roberta-base-sentiment-latest).

Multi-teacher distillation: Codex stays the teacher for our real posts; this English Twitter model teaches
English social text (slang, sarcasm, memes), where the benchmark showed it is 5-26 points better than our student.
It only sees English / Latin-script text (it is English-only): Hindi/Marathi posts and romanized Hinglish are skipped.

Inputs (whatever exists under data/):
  silver/llm_codex.jsonl + silver/teacher_labels.jsonl   the Codex-labelled real posts (English ones)
  silver/extra_english.jsonl                             extra unlabelled English social posts
  raw/tweeteval/train_text.txt                           TweetEval train tweets
Output: silver/roberta_soft.jsonl  {"key": normalised text, "text": original, "probs": [neg, neu, pos], "src": ...}

Run on a GPU (Colab T4: ~5 min for ~90k texts). On CPU it works but takes hours.
"""
import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.preprocessing.post_text import _norm  # noqa: E402

TEACHER = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def is_english_like(text):
    from src.serving.cascade import is_romanized_indic, non_latin_share
    return non_latin_share(text) <= 0.3 and not is_romanized_indic(text)


def collect(limit_extra):
    silver = os.path.join(data_root, "data", "silver")
    texts = []
    codex_path = os.path.join(silver, "llm_codex.jsonl")
    if os.path.exists(codex_path):
        ids = {json.loads(l)["id"] for l in open(codex_path, encoding="utf-8")}
        for line in open(os.path.join(silver, "teacher_labels.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in ids and is_english_like(r["text"]):
                texts.append((r["text"], "codex_post"))
    extra = os.path.join(silver, "extra_english.jsonl")
    if os.path.exists(extra):
        rows = [json.loads(l)["text"] for l in open(extra, encoding="utf-8")]
        texts += [(t, "extra_english") for t in rows[: limit_extra or None]]
    te = os.path.join(data_root, "data", "raw", "tweeteval", "train_text.txt")
    if os.path.exists(te):
        texts += [(l.strip(), "tweeteval_train") for l in open(te, encoding="utf-8")]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit-extra", type=int, default=0, help="cap extra English posts (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="cap total texts (smoke tests)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER).to(device).eval()
    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert sorted(order) == ["negative", "neutral", "positive"], order
    perm = [order.index(l) for l in ("negative", "neutral", "positive")]

    texts = collect(args.limit_extra)
    if args.limit:
        texts = texts[: args.limit]
    seen, uniq = set(), []
    for t, src in texts:
        k = _norm(t)
        if k and k not in seen:
            seen.add(k)
            uniq.append((t, src, k))
    print(f"teacher: {TEACHER} on {device}; {len(uniq):,} unique texts", flush=True)

    out_path = os.path.join(data_root, "data", "silver", "roberta_soft.jsonl")
    uniq.sort(key=lambda x: len(x[0]))
    with open(out_path, "w", encoding="utf-8") as out, torch.inference_mode():
        for i in range(0, len(uniq), args.batch):
            chunk = uniq[i: i + args.batch]
            enc = tok([t for t, _, _ in chunk], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), -1).cpu().tolist()
            for (t, src, k), p in zip(chunk, probs):
                out.write(json.dumps({"key": k, "text": t, "probs": [round(p[j], 4) for j in perm], "src": src},
                                     ensure_ascii=False) + "\n")
            if (i // args.batch) % 100 == 0:
                print(f"  {i + len(chunk):,}/{len(uniq):,}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
