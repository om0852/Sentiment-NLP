"""Step 3 (plus the student half of step 5) of the accuracy plan.

Fine-tunes microsoft/Multilingual-MiniLM-L12-H384 on:
  * silver posts from scripts/build_silver_dataset.py where the local teacher AND the stored DB label agree
    (distilled with the teacher's soft probabilities), and
  * TweetEval train + val (human labels, English slang). The TweetEval test split stays held out.

Then trims the 250k-token multilingual vocabulary to the tokens our data actually uses, exports to ONNX,
quantizes to int8, fits a softmax temperature on a held-out calibration split, and writes everything the
serving runtime needs to models/student/.

Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/train_student.py --epochs 2
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.models.calibration import fit_temperature  # noqa: E402
from src.preprocessing.post_text import post_to_text  # noqa: E402

STUDENT = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER_SRC = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # same XLM-R sentencepiece vocab
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT_DIR = os.path.join(project_root, "models", "student")


def bucket(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100


def load_data():
    silver_path = os.path.join(data_root, "data", "silver", "teacher_labels.jsonl")
    train, val, calib = [], [], []
    disagree = 0
    with open(silver_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["original"] not in L2I:
                continue
            if r["teacher"] != r["original"]:
                disagree += 1
                continue
            item = {"text": r["text"], "label": L2I[r["teacher"]], "soft": r["probs"], "src": "silver"}
            b = bucket(r["id"])
            (val if b < 4 else calib if b < 8 else train).append(item)
    print(f"silver agree: train {len(train):,} val {len(val):,} calib {len(calib):,}; "
          f"dropped {disagree:,} posts where teacher and DB label disagree", flush=True)

    te_dir = os.path.join(data_root, "data", "raw", "tweeteval")
    te_map = {0: 0, 1: 1, 2: 2}  # TweetEval ids already use negative/neutral/positive order
    for split, dest in [("train", train), ("val", calib)]:
        with open(os.path.join(te_dir, f"{split}_text.txt"), encoding="utf-8") as ft, \
                open(os.path.join(te_dir, f"{split}_labels.txt"), encoding="utf-8") as fl:
            for t, l in zip(ft, fl):
                dest.append({"text": t.strip(), "label": te_map[int(l)], "soft": None, "src": "tweeteval"})
    print(f"with TweetEval: train {len(train):,} val {len(val):,} calib {len(calib):,}", flush=True)
    return train, val, calib


def corpus_texts(gold_ids):
    """All non-gold post texts plus TweetEval train/val, used only to decide which tokens to keep."""
    with open(os.path.join(data_root, "data", "raw", "posts_300k.json"), encoding="utf-8") as f:
        for p in json.load(f):
            if p["id"] not in gold_ids:
                yield post_to_text(p.get("title"), p.get("description"))
    te_dir = os.path.join(data_root, "data", "raw", "tweeteval")
    for split in ["train", "val"]:
        with open(os.path.join(te_dir, f"{split}_text.txt"), encoding="utf-8") as f:
            yield from (line.strip() for line in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--vocab", type=int, default=60000, help="max kept tokens after trimming")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of soft teacher loss for silver data")
    ap.add_argument("--max-train", type=int, default=0, help="cap training rows (0 = all)")
    ap.add_argument("--layers", type=int, default=6, help="transformer layers to keep (MiniLM has 12)")
    ap.add_argument("--resume", default="", help="state_dict to start from (same --layers/--max-len/--vocab)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(OUT_DIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    train, val, calib = load_data()
    if args.max_train:
        random.shuffle(train)
        train = train[: args.max_train]

    # ---- vocab trimming: keep specials + the most frequent tokens in our corpus ----
    with open(os.path.join(data_root, "data", "gold", "gold_ids.json")) as f:
        gold_ids = set(json.load(f))
    # cache key covers everything that changes the counts: tokenizer, max_len and which posts are excluded
    key = hashlib.md5(f"{TOKENIZER_SRC}|{args.max_len}|{sorted(gold_ids)}".encode()).hexdigest()[:10]
    counts_cache = os.path.join(data_root, "data", "silver", f"token_counts_{key}.json")
    t0 = time.time()
    if os.path.exists(counts_cache):
        with open(counts_cache) as f:
            counts = collections.Counter({int(k): v for k, v in json.load(f).items()})
    else:
        counts = collections.Counter()
        batch = []
        for text in corpus_texts(gold_ids):
            batch.append(text)
            if len(batch) == 2000:
                for ids in tok(batch, truncation=True, max_length=args.max_len,
                               add_special_tokens=False)["input_ids"]:
                    counts.update(ids)
                batch = []
        if batch:
            for ids in tok(batch, truncation=True, max_length=args.max_len, add_special_tokens=False)["input_ids"]:
                counts.update(ids)
        with open(counts_cache, "w") as f:
            json.dump(counts, f)
    specials = sorted(set(tok.all_special_ids))
    keep = specials + [i for i, _ in counts.most_common(args.vocab) if i not in specials]
    old_vocab = len(tok)
    vocab_map = np.full(old_vocab, keep.index(tok.unk_token_id), dtype=np.int32)
    for new_id, old_id in enumerate(keep):
        vocab_map[old_id] = new_id
    covered = sum(c for i, c in counts.items() if vocab_map[i] != vocab_map[tok.unk_token_id] or i == tok.unk_token_id)
    print(f"vocab trimmed {old_vocab:,} -> {len(keep):,} tokens, covering "
          f"{covered / sum(counts.values()):.2%} of token occurrences ({time.time() - t0:.0f}s)", flush=True)
    # Trimmed tokenizer: the same Unigram model restricted to the kept pieces, ids in `keep` order.
    # The full 250k-piece tokenizer costs ~260 MB of RAM at serving time; this one ~50 MB. Training uses
    # it too, so the model sees exactly the segmentation it will see in production.
    from tokenizers import Tokenizer
    tj = json.loads(tok.backend_tokenizer.to_str())
    full_vocab = tj["model"]["vocab"]
    tj["model"]["vocab"] = [full_vocab[o] for o in keep]
    tj["model"]["unk_id"] = int(vocab_map[tok.unk_token_id])
    for a in tj["added_tokens"]:
        a["id"] = int(vocab_map[a["id"]])
    for spec in tj["post_processor"].get("special_tokens", {}).values():
        spec["ids"] = [int(vocab_map[i]) for i in spec["ids"]]
    trimmed_tok = Tokenizer.from_str(json.dumps(tj, ensure_ascii=False))
    pad_id = int(vocab_map[tok.pad_token_id])
    # parity check: trimmed ids must stay inside the trimmed embedding table, and specials must map 1:1
    probe = trimmed_tok.encode("Check खूप छान 😂 <mask> test").ids
    assert max(probe) < len(keep), "trimmed tokenizer produced an id outside the trimmed vocabulary"
    for special in ("<s>", "</s>", "<pad>", "<unk>"):
        assert trimmed_tok.token_to_id(special) == int(vocab_map[tok.convert_tokens_to_ids(special)])
    trimmed_tok.enable_truncation(max_length=args.max_len)
    trimmed_tok.enable_padding(pad_id=pad_id, pad_token="<pad>")

    model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=3)
    if args.layers and args.layers < model.config.num_hidden_layers:
        # keep evenly spaced layers (DistilBERT-style init): 12 -> 6 keeps layers 0, 2, 4, 6, 8, 10
        n = model.config.num_hidden_layers
        kept = [round(i * n / args.layers) for i in range(args.layers)]
        model.bert.encoder.layer = torch.nn.ModuleList([model.bert.encoder.layer[i] for i in kept])
        model.config.num_hidden_layers = args.layers
        print(f"student layers: kept {kept} of {n}", flush=True)
    emb = model.get_input_embeddings()
    new_emb = torch.nn.Embedding(len(keep), emb.weight.shape[1])
    new_emb.weight.data = emb.weight.data[torch.tensor(keep)].clone()
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(keep)
    model.config.pad_token_id = pad_id

    def encode(items):
        enc = trimmed_tok.encode_batch([x["text"] if x["text"].strip() else "." for x in items])
        ids = torch.tensor([e.ids for e in enc], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long)
        return ids, mask

    def predict_logits(items, bs=128):
        model.eval()
        out = []
        with torch.inference_mode():
            order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
            for i in range(0, len(order), bs):
                chunk = [items[j] for j in order[i: i + bs]]
                ids, mask = encode(chunk)
                out.append(model(input_ids=ids, attention_mask=mask).logits)
        logits = torch.cat(out)
        res = torch.empty_like(logits)
        res[torch.tensor(order)] = logits
        return res

    def accuracy(items):
        logits = predict_logits(items)
        gold = torch.tensor([x["label"] for x in items])
        return (logits.argmax(-1) == gold).float().mean().item()

    if args.resume:
        # continue from (or just export) a checkpoint saved by an earlier run with the same settings
        model.load_state_dict(torch.load(args.resume))
        print(f"resumed weights from {args.resume}", flush=True)

    # ---- training ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    print(f"training {total:,} steps ({args.epochs} epochs x {steps_per_epoch:,})", flush=True)

    step, best, t0 = 0, -1.0, time.time()
    while step < total:
        # length-bucketed shuffling: shuffle, then sort within chunks of 50 batches to cut padding
        random.shuffle(train)
        chunk = args.batch * 50
        order = []
        for i in range(0, len(train), chunk):
            order += sorted(train[i: i + chunk], key=lambda x: len(x["text"]))
        batches = [order[i: i + args.batch] for i in range(0, len(order), args.batch)]
        random.shuffle(batches)
        for b in batches:
            if step >= total:
                break
            model.train()
            ids, mask = encode(b)
            logits = model(input_ids=ids, attention_mask=mask).logits
            hard = torch.tensor([x["label"] for x in b])
            loss = F.cross_entropy(logits, hard)
            soft_rows = [i for i, x in enumerate(b) if x["soft"] is not None]
            if soft_rows:
                soft = torch.tensor([b[i]["soft"] for i in soft_rows])
                kl = F.kl_div(F.log_softmax(logits[soft_rows], -1), soft, reduction="batchmean")
                loss = (1 - args.alpha) * loss + args.alpha * kl * len(soft_rows) / len(b) + \
                    args.alpha * F.cross_entropy(logits, hard) * (1 - len(soft_rows) / len(b))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"step {step:,}/{total:,} loss {loss.item():.3f} "
                      f"eta {(total - step) * el / step / 60:.0f} min", flush=True)
            if step % 1500 == 0 or step == total:
                acc = accuracy(val)
                print(f"  val accuracy (silver agree set) {acc:.2%}", flush=True)
                if acc > best:
                    best = acc
                    torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
    if best < 0:  # no training steps (e.g. --epochs 0 with --resume): export the weights as they are
        best = accuracy(val)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
    print(f"best val accuracy {best:.2%}", flush=True)

    # ---- calibration (step 5): temperature on held-out calib split ----
    calib_logits = predict_logits(calib).numpy()
    temperature = fit_temperature(calib_logits, np.array([x["label"] for x in calib]))
    print(f"fitted temperature {temperature:.3f}", flush=True)

    # ---- export: ONNX fp32 -> dynamic int8 ----
    model.eval()
    fp32 = os.path.join(OUT_DIR, "model.fp32.onnx")
    dummy_ids = torch.ones(2, 16, dtype=torch.long)
    dummy_mask = torch.ones(2, 16, dtype=torch.long)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.onnx.export(Wrap(model), (dummy_ids, dummy_mask), fp32, input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                    "logits": {0: "b"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8 = os.path.join(OUT_DIR, "model.int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    os.remove(fp32)

    trimmed_tok.no_truncation()
    trimmed_tok.no_padding()
    trimmed_tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
    stale_map = os.path.join(OUT_DIR, "vocab_map.npy")
    if os.path.exists(stale_map):
        os.remove(stale_map)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"labels": LABELS, "max_len": args.max_len, "temperature": round(float(temperature), 4),
                   "base_model": STUDENT, "layers": model.config.num_hidden_layers, "teacher": TOKENIZER_SRC, "kept_vocab": len(keep),
                   "val_accuracy_silver": round(best, 4)}, f, indent=2)
    os.remove(os.path.join(OUT_DIR, "best.pt"))
    print(f"saved -> {OUT_DIR} (int8 model {os.path.getsize(int8) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
