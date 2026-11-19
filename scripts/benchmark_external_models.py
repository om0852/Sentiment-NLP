"""Benchmark our engine against off-the-shelf Hugging Face sentiment models on the same held-out sets.

Models compared:
  ours (hybrid)       the shipped engine (TF-IDF + rules + int8 ONNX transformer), via classify_batch
  ours (transformer)  the transformer alone
  cardiffnlp/twitter-roberta-base-sentiment-latest   English Twitter RoBERTa (3 classes)
  tabularisai/multilingual-sentiment-analysis        multilingual DistilBERT (5 classes, mapped to 3)

Every model gets the same cleaned text (src/preprocessing/post_text.post_to_text) and the same scoring.

Sets (whatever exists): real posts labelled per the guide (--extra files), Codex-labelled held-out real posts
(val split, never trained on), TweetEval test, hard sets.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/benchmark_external_models.py \
        --extra path/to/claude_gold200.py --extra path/to/realworld_set.py --json-out bench.json
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from evaluate_cascade import Engine, MemoStudent, load_extra, load_hard, load_tweeteval  # noqa: E402
from src.preprocessing.context_analyzer import ContextAnalyzer  # noqa: E402
from src.preprocessing.post_text import post_to_text  # noqa: E402
from src.serving.cascade import load_config, non_latin_share  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
FIVE_TO_THREE = {"very negative": "negative", "negative": "negative", "neutral": "neutral",
                 "positive": "positive", "very positive": "positive"}
EXTERNAL = {
    "roberta-latest (EN)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "tabularisai (multi)": "tabularisai/multilingual-sentiment-analysis",
}


def bucket(k):
    return int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16) % 100


def codex_val():
    path = os.path.join(data_root, "data", "silver", "llm_codex.jsonl")
    if not os.path.exists(path):
        return []
    codex = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(path, encoding="utf-8")}
    out = []
    with open(os.path.join(data_root, "data", "silver", "teacher_labels.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in codex and codex[r["id"]] in LABELS and bucket(r["id"]) < 4:
                out.append({"text": r["text"], "ok": {codex[r["id"]]}, "strict": codex[r["id"]], "cat": r["source"]})
    return out


class HFModel:
    def __init__(self, repo, threads):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def predict(self, texts, bs=32, max_len=128):
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [None] * len(texts)
        with self.torch.inference_mode():
            for i in range(0, len(order), bs):
                idx = order[i: i + bs]
                enc = self.tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                               padding=True, return_tensors="pt")
                for j, k in zip(idx, self.model(**enc).logits.argmax(-1).tolist()):
                    lab = self.id2label[k]
                    res[j] = FIVE_TO_THREE.get(lab, lab)
        return res


def acc(items, labels):
    hit = [(lab == x["strict"]) if "strict" in x else (lab in x["ok"]) for x, lab in zip(items, labels)]
    return sum(hit) / len(hit) if hit else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--tweeteval-n", type=int, default=0, help="0 = full test set")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--cache", default="", help="json file caching predictions per model, so reruns skip finished ones")
    ap.add_argument("--model-path", action="append", default=[],
                    help="override where a model loads from, e.g. tabularisai=/local/dir (for converted weights)")
    args = ap.parse_args()

    sets = {}
    for p in args.extra:
        sets[os.path.basename(p)[:-3]] = load_extra(p)
    sets["codex_val (real posts)"] = codex_val()
    te = load_tweeteval()
    sets["tweeteval_test"] = te[: args.tweeteval_n] if args.tweeteval_n else te
    sets["hard_1"] = load_hard("benchmark_hard_50.py")
    sets["hard_2"] = load_hard("benchmark_hard_set_2.py")
    sets = {k: v for k, v in sets.items() if v}
    for items in sets.values():
        for x in items:
            x["clean"] = post_to_text(x["text"]) or x["text"]

    preds = collections.defaultdict(dict)
    speed = {}
    # cache is only reused when every set's texts and labels are identical
    fingerprint = hashlib.md5(json.dumps({k: [[x["text"], sorted(x["ok"])] for x in v] for k, v in sets.items()},
                                         ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if args.cache and os.path.exists(args.cache):
        cached = json.load(open(args.cache, encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            for m, by_set in cached.get("preds", {}).items():
                if all(k in by_set and len(by_set[k]) == len(v) for k, v in sets.items()):
                    preds[m] = by_set
        speed.update(cached.get("speed", {}))
        print("cached models:", list(preds), flush=True)

    def save_cache():
        if args.cache:
            json.dump({"preds": preds, "speed": speed, "fingerprint": fingerprint},
                      open(args.cache, "w", encoding="utf-8"))
    overrides = dict(o.split("=", 1) for o in args.model_path)

    # ours
    eng = Engine()
    sample = [x["clean"] for x in sets.get("codex_val (real posts)", sets[next(iter(sets))])][:300]
    cfg = load_config()
    an = ContextAnalyzer(disabled_rules=cfg.get("disabled_rules", []))
    only = dict(cfg, escalate_below=1.01, short_text_chars=0, keep_romanized_indic_on_tier1=False,
                rules_override_student=False)
    if "ours (hybrid)" not in preds:
        for name, items in sets.items():
            preds["ours (hybrid)"][name] = [r["label"] for r in eng.run(items, cfg, an)]
            preds["ours (transformer)"][name] = [r["label"] for r in eng.run(items, only, an)]
            print(f"ours done: {name}", flush=True)
        raw = eng.student.student if isinstance(eng.student, MemoStudent) else eng.student
        t = time.process_time(); raw.predict_batch(sample)
        speed["ours (transformer)"] = (time.process_time() - t) * 1000 / len(sample)
        save_cache()

    for short, repo in EXTERNAL.items():
        if short in preds:
            continue
        repo = next((v for k, v in overrides.items() if k in repo), repo)
        m = HFModel(repo, args.threads)
        for name, items in sets.items():
            preds[short][name] = m.predict([x["clean"] for x in items])
            print(f"{short} done: {name}", flush=True)
        m1 = HFModel(repo, 1)
        t = time.process_time(); m1.predict(sample, bs=16); speed[short] = (time.process_time() - t) * 1000 / len(sample)
        save_cache()

    models = list(preds)
    print("\nAccuracy (3-class) on held-out sets")
    print(f"{'set':28} {'n':>6} | " + " | ".join(f"{m:>19}" for m in models))
    report = {"sets": {}, "speed_ms_per_post_1thread": speed}
    for name, items in sets.items():
        row = {m: acc(items, preds[m][name]) for m in models}
        report["sets"][name] = row
        print(f"{name:28} {len(items):6} | " + " | ".join(f"{row[m]:19.1%}" for m in models))
        if name.startswith(("claude", "codex")):
            for script, sel in (("  ↳ Latin script", lambda x: non_latin_share(x["text"]) <= 0.3),
                                ("  ↳ Devanagari/other", lambda x: non_latin_share(x["text"]) > 0.3)):
                idx = [i for i, x in enumerate(items) if sel(x)]
                sub = [items[i] for i in idx]
                r = {m: acc(sub, [preds[m][name][i] for i in idx]) for m in models}
                report["sets"][name + script] = r
                print(f"{script:28} {len(sub):6} | " + " | ".join(f"{r[m]:19.1%}" for m in models))
    print(f"{'model CPU ms/post (1 thread)':28} {'':6} | " + " | ".join(
        f"{speed.get(m, speed.get('ours (transformer)')):19.1f}" for m in models))
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
