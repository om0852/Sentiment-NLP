"""Serving runtime for the distilled multilingual student (step 3).

Needs only onnxruntime, tokenizers and numpy: no torch or transformers at serving time.
Files (written by scripts/train_student.py) live in models/student/:
    model.int8.onnx   quantized MiniLM-L12-H384 with a trimmed vocabulary
    tokenizer.json    XLM-R sentencepiece (Unigram) tokenizer trimmed to the same ~60k pieces, so its ids
                      index the model's embeddings directly (the full tokenizer would cost ~260 MB of RAM)
    config.json       labels, max_len, calibrated temperature
"""
import json
import os
from typing import Dict, List

import numpy as np

from src.models.calibration import softmax


def _parts(model_dir: str) -> List[str]:
    """Large models are stored as model.int8.onnx.part00, part01, ... (GitHub's 100 MB per-file limit)."""
    import glob
    return sorted(glob.glob(os.path.join(model_dir, "model.int8.onnx.part*")))


def _model_source(model_dir: str):
    path = os.path.join(model_dir, "model.int8.onnx")
    if os.path.exists(path):
        return path
    parts = _parts(model_dir)
    expected = [os.path.join(model_dir, f"model.int8.onnx.part{i:02d}") for i in range(len(parts))]
    if not parts or [os.path.normcase(p) for p in parts] != [os.path.normcase(p) for p in expected]:
        raise FileNotFoundError(f"model parts in {model_dir} are missing or not numbered part00..: {parts}")
    data = bytearray()
    for part in parts:
        with open(part, "rb") as f:
            chunk = f.read()
        if not chunk:
            raise ValueError(f"empty model part {part}")
        data += chunk
    return bytes(data)


class StudentClassifier:
    def __init__(self, model_dir: str, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.labels: List[str] = cfg["labels"]
        self.max_len: int = cfg["max_len"]
        self.temperature: float = cfg.get("temperature", 1.0)

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_len)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False  # the arena keeps ~40 MB of peak buffers around; we're RAM-bound
        self.session = ort.InferenceSession(_model_source(model_dir), opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def available(model_dir: str) -> bool:
        return os.path.exists(os.path.join(model_dir, "model.int8.onnx")) or bool(_parts(model_dir))

    def logits(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        enc = self.tokenizer.encode_batch([t if t.strip() else "." for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        out: List[Dict] = []
        for i in range(0, len(texts), batch_size):
            probs = softmax(self.logits(texts[i: i + batch_size]), self.temperature)
            for p in probs:
                k = int(p.argmax())
                out.append({
                    "label": self.labels[k],
                    "confidence": float(p[k]),
                    "probabilities": {l: round(float(v), 4) for l, v in zip(self.labels, p)},
                })
        return out
