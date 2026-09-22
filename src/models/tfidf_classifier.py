import os
import joblib
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseSentimentClassifier

class TfidfSentimentClassifier(BaseSentimentClassifier):
    def __init__(self, max_features: int = 15000, confidence_threshold: float = 0.60):
        self.max_features = max_features
        self.confidence_threshold = confidence_threshold
        self.classes_ = ["negative", "neutral", "positive"]
        
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=2,
                token_pattern=r"(?u)\b\w+\b|_[a-z0-9_]+_"  # capture semantic tokens like _emoji_fire_
            )),
            ("clf", LogisticRegression(
                C=1.5,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced"
            ))
        ])
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        """Trains the TF-IDF and calibrated classifier."""
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        self.is_trained = True

    def predict(self, text: str, confidence_threshold: float = None) -> Dict[str, Any]:
        """Inference for a single text."""
        res = self.predict_batch([text], confidence_threshold=confidence_threshold)
        return res[0]

    def predict_batch(self, texts: List[str], confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """High-speed batch inference."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before running prediction.")

        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Probabilities matrix: shape (N, n_classes)
        prob_matrix = self.pipeline.predict_proba(texts)
        
        results = []
        for i, text in enumerate(texts):
            probs = prob_matrix[i]
            max_idx = int(np.argmax(probs))
            predicted_label = self.classes_[max_idx]
            confidence = float(probs[max_idx])
            
            prob_dict = {cls: float(probs[j]) for j, cls in enumerate(self.classes_)}
            
            # Sarcasm flag or low-confidence check
            has_sarcasm_marker = "_sarcasm_tag_" in text
            fallback_required = (confidence < thresh) or has_sarcasm_marker
            
            results.append({
                "text": text,
                "label": predicted_label,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                "fallback_required": fallback_required,
                "reason": "low_confidence" if (confidence < thresh) else ("sarcasm_detected" if has_sarcasm_marker else None)
            })

        return results

    def save(self, path: str):
        """Saves model to disk with zlib compression."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "classes_": self.classes_,
            "max_features": self.max_features,
            "confidence_threshold": self.confidence_threshold,
            "is_trained": self.is_trained
        }, path, compress=3)

    def load(self, path: str):
        """Loads serialized model."""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.classes_ = data["classes_"]
        self.max_features = data.get("max_features", 15000)
        self.confidence_threshold = data.get("confidence_threshold", 0.60)
        self.is_trained = data.get("is_trained", True)
