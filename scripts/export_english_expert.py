"""Export cardiffnlp/twitter-roberta-base-sentiment-latest as the "English expert" (int8 ONNX).

The router sends short English social posts to this model (see src/serving/cascade.py). It is English-only, so
it never sees Hindi/Marathi, romanized Hinglish, news or long posts.

Writes models/english_expert/:
  model.int8.onnx.partNN   int8 ONNX split into <50 MB parts (GitHub rejects files over 100 MB; no LFS needed)
  tokenizer.json           fast BPE tokenizer
  config.json              labels, max_len, temperature (fitted on TweetEval val, human labels)

Usage: SENTIMENT_DATA_ROOT=/checkout/with/data python scripts/export_english_expert.py
"""
import glob
import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.models.calibration import expected_calibration_error, fit_temperature, softmax  # noqa: E402

REPO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUT = os.path.join(project_root, "models", "english_expert")
LABELS = ["negative", "neutral", "positive"]
PART_BYTES = 45 * 1024 * 1024
MAX_LEN = 128


def main():
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert order == LABELS, f"unexpected label order {order}"

    # temperature on TweetEval val (human labels)
    te = os.path.join(data_root, "data", "raw", "tweeteval")
    texts = [l.strip() for l in open(os.path.join(te, "val_text.txt"), encoding="utf-8")]
    gold = np.array([int(l) for l in open(os.path.join(te, "val_labels.txt"))])
    logits = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i: i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            logits.append(model(**enc).logits.numpy())
    logits = np.concatenate(logits)
    t = fit_temperature(logits, gold)
    print(f"TweetEval val accuracy {(logits.argmax(1) == gold).mean():.1%}; temperature {t:.3f}; "
          f"ECE {expected_calibration_error(softmax(logits), gold):.3f} -> "
          f"{expected_calibration_error(softmax(logits, t), gold):.3f}", flush=True)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    fp32 = os.path.join(OUT, "model.fp32.onnx")
    dummy = torch.ones(2, 16, dtype=torch.long)
    torch.onnx.export(Wrap(model), (dummy, dummy), fp32, input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                    "logits": {0: "b"}})
    int8 = os.path.join(OUT, "model.int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    os.remove(fp32)

    for old in glob.glob(int8 + ".part*"):
        os.remove(old)
    data = open(int8, "rb").read()
    for n, i in enumerate(range(0, len(data), PART_BYTES)):
        with open(f"{int8}.part{n:02d}", "wb") as f:
            f.write(data[i: i + PART_BYTES])
    os.remove(int8)

    tok.backend_tokenizer.save(os.path.join(OUT, "tokenizer.json"))
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({"labels": LABELS, "max_len": MAX_LEN, "temperature": round(float(t), 4), "source": REPO,
                   "pad_token": tok.pad_token, "note": "English-only; used for short English social posts"}, f, indent=2)
    print(f"saved -> {OUT} ({len(data) / 1e6:.1f} MB int8 in {len(glob.glob(int8 + '.part*'))} parts)")


if __name__ == "__main__":
    main()
