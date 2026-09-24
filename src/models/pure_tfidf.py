import os
import sys
import gzip
import json
import math
import re
import random
from collections import Counter
from typing import List, Dict, Any, Optional

class PureSentimentClassifier:
    """
    100% Pure Python Multilingual Classifier (English, Hindi, Marathi).
    Uses Sublinear TF-IDF with L2 Document Normalization and SGD Softmax Optimization.
    Immune to cross-language text length biases and Windows WDAC binary policies.
    Sub-millisecond inference and sub-25MB RAM footprint.
    """
    def __init__(self, max_features: int = 40000, lr: float = 0.4, epochs: int = 10, l2_reg: float = 1e-5):
        self.max_features = max_features
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.classes_ = ["negative", "neutral", "positive"]
        self.vocab_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self.weights_: Dict[str, Dict[str, float]] = {c: {} for c in self.classes_}
        self.biases_: Dict[str, float] = {c: 0.0 for c in self.classes_}
        self.is_trained = False

    def _tokenize(self, text: str) -> List[str]:
        # Capture Indic Devanagari script, Latin words, and semantic tokens like _emoji_fire_
        words = re.findall(r"[\u0900-\u097F]+|\b\w+\b|_[a-z0-9_]+_", text.lower())
        tokens = list(words)
        # Add bigrams for local context (negations, idioms, compound phrases)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _get_tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        vec: Dict[str, float] = {}
        sq_sum = 0.0
        for t, cnt in counts.items():
            if t in self.vocab_:
                # Sublinear TF scaling
                val = (1.0 + math.log(cnt)) * self.idf_[t]
                vec[t] = val
                sq_sum += val * val
        if sq_sum > 0:
            norm = math.sqrt(sq_sum)
            for t in vec:
                vec[t] /= norm
        return vec

    def fit(self, texts: List[str], labels: List[str]):
        n_samples = len(texts)
        doc_freq = Counter()
        tokenized_docs = []
        
        # 1. Build Document Frequencies
        for t in texts:
            toks = self._tokenize(t)
            tokenized_docs.append(toks)
            doc_freq.update(set(toks))

        # 2. Select Vocabulary & Compute Smooth IDF
        min_df = 2 if n_samples > 100 else 1
        most_common = [w for w, cnt in doc_freq.most_common(self.max_features) if cnt >= min_df]
        self.vocab_ = {w: i for i, w in enumerate(most_common)}
        self.idf_ = {w: math.log((1.0 + n_samples) / (1.0 + doc_freq[w])) + 1.0 for w in self.vocab_}

        for c in self.classes_:
            self.weights_[c] = {w: 0.0 for w in self.vocab_}
            self.biases_[c] = 0.0

        # Precompute sparse TF-IDF vectors
        sparse_vecs = [self._get_tfidf_vec(toks) for toks in tokenized_docs]
        dataset = list(zip(sparse_vecs, labels))

        # 3. Train Softmax Regression via SGD with L2 Weight Decay
        for epoch in range(self.epochs):
            random.seed(42 + epoch)
            random.shuffle(dataset)
            cur_lr = self.lr / (1.0 + 0.12 * epoch)

            for vec, target_lbl in dataset:
                if not vec or target_lbl not in self.classes_:
                    continue

                # Forward pass: linear combination
                scores = {}
                for c in self.classes_:
                    s = self.biases_[c]
                    w_c = self.weights_[c]
                    for feat, val in vec.items():
                        s += w_c[feat] * val
                    scores[c] = s

                # Softmax probabilities with numerical stability clamp
                max_s = max(scores.values())
                exp_s = {c: math.exp(max(-20.0, min(20.0, s - max_s))) for c, s in scores.items()}
                sum_exp = sum(exp_s.values())
                probs = {c: exp_s[c] / sum_exp for c in self.classes_}

                # Backward pass: gradient descent on cross-entropy loss
                for c in self.classes_:
                    grad = probs[c] - (1.0 if c == target_lbl else 0.0)
                    self.biases_[c] -= cur_lr * grad
                    w_c = self.weights_[c]
                    for feat, val in vec.items():
                        # Sparse SGD update with L2 weight regularization
                        w_c[feat] -= cur_lr * (grad * val + self.l2_reg * w_c[feat])

        self.is_trained = True
        return self

    def predict_one(self, text: str) -> Dict[str, Any]:
        toks = self._tokenize(text)
        vec = self._get_tfidf_vec(toks)
        scores = {}
        for c in self.classes_:
            s = self.biases_[c]
            w_c = self.weights_[c]
            for feat, val in vec.items():
                s += w_c.get(feat, 0.0) * val
            scores[c] = s

        max_s = max(scores.values()) if scores else 0.0
        exp_s = {c: math.exp(max(-20.0, min(20.0, s - max_s))) for c, s in scores.items()}
        sum_exp = sum(exp_s.values()) if exp_s else 1.0
        probs = {c: exp_s[c] / sum_exp for c in self.classes_}

        best = max(probs, key=probs.get)
        return {
            "label": best,
            "confidence": probs[best],
            "probabilities": probs
        }

    def predict_batch(self, texts: List[str], confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        return [self.predict_one(t) for t in texts]

    def save(self, filepath: str):
        # Save compact non-zero weights
        compact_weights = {}
        for c in self.classes_:
            compact_weights[c] = {k: round(v, 5) for k, v in self.weights_[c].items() if abs(v) > 1e-4}

        compact_idf = {k: round(v, 4) for k, v in self.idf_.items()}

        data = {
            "max_features": self.max_features,
            "classes_": self.classes_,
            "vocab_": self.vocab_,
            "idf_": compact_idf,
            "weights_": compact_weights,
            "biases_": self.biases_
        }
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def load(self, filepath: str):
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.max_features = data.get("max_features", 40000)
        self.classes_ = data.get("classes_", ["negative", "neutral", "positive"])
        self.vocab_ = data["vocab_"]
        self.idf_ = data.get("idf_", {})
        self.weights_ = data["weights_"]
        self.biases_ = data.get("biases_", {c: 0.0 for c in self.classes_})
        self.is_trained = True
        return self

# Standard Aliases
PureLogOddsClassifier = PureSentimentClassifier
TfidfSentimentClassifier = PureSentimentClassifier
TFIDFClassifier = PureSentimentClassifier
