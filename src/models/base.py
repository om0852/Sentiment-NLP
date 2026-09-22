from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSentimentClassifier(ABC):
    @abstractmethod
    def train(self, texts: List[str], labels: List[str]):
        """Trains the model on preprocessed texts and corresponding labels."""
        pass

    @abstractmethod
    def predict(self, text: str, confidence_threshold: float = 0.60) -> Dict[str, Any]:
        """
        Predicts sentiment for a single text.
        Returns: {
            'label': 'positive' | 'negative' | 'neutral',
            'confidence': float,
            'probabilities': { 'positive': float, 'negative': float, 'neutral': float },
            'fallback_required': bool
        }
        """
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str], confidence_threshold: float = 0.60) -> List[Dict[str, Any]]:
        """Batch prediction over list of texts."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Serializes model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads serialized model from disk."""
        pass
