import os
import json
import time
from typing import Dict, Any, Optional

class ActiveLearningQueue:
    """
    Appends low-confidence (<0.60), ambiguous, or fallback posts to a local
    feedback log for continuous active learning and weekly retraining.
    """
    def __init__(self, log_path: str = None):
        if log_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_path = os.path.join(project_root, "data", "feedback_queue.jsonl")
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def log_feedback(self, text: str, initial_label: str, confidence: float, reason: str, resolved_label: Optional[str] = None):
        entry = {
            "timestamp": time.time(),
            "text": text,
            "initial_label": initial_label,
            "confidence": round(confidence, 4),
            "reason": reason,
            "resolved_label": resolved_label
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Non-blocking

    def count_queued(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

# Aliases for compatibility
ActiveLearningBuffer = ActiveLearningQueue
