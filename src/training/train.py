import os
import sys
import json
import time
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.tfidf_classifier import TfidfSentimentClassifier

def load_split(split_name: str, data_dir: str):
    file_path = os.path.join(data_dir, f"{split_name}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_training(data_dir: str = None, model_output_path: str = None) -> Dict[str, Any]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data", "processed")
    if model_output_path is None:
        model_output_path = os.path.join(project_root, "models", "sentiment_model.joblib")

    print(f"Loading data from: {data_dir}", flush=True)
    train_data = load_split("train", data_dir)
    val_data = load_split("val", data_dir)
    test_data = load_split("test", data_dir)

    print(f"Dataset split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}", flush=True)

    # Extract texts and labels
    train_texts = [d.get("processed_text", d.get("raw_text", "")) for d in train_data]
    train_labels = [d["label"].lower().strip() for d in train_data]

    test_texts = [d.get("processed_text", d.get("raw_text", "")) for d in test_data]
    test_labels = [d["label"].lower().strip() for d in test_data]

    print(f"\nInitializing TfidfSentimentClassifier...", flush=True)
    model = TfidfSentimentClassifier(max_features=15000, confidence_threshold=0.60)

    start_train_time = time.perf_counter()
    model.train(train_texts, train_labels)
    train_duration = time.perf_counter() - start_train_time
    print(f"Training completed in {train_duration:.3f} seconds!", flush=True)

    # Evaluate on test set
    start_eval_time = time.perf_counter()
    predictions = model.predict_batch(test_texts)
    eval_duration = time.perf_counter() - start_eval_time

    pred_labels = [p["label"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    fallback_flags = [p["fallback_required"] for p in predictions]

    acc = accuracy_score(test_labels, pred_labels)
    macro_f1 = f1_score(test_labels, pred_labels, average="macro")
    avg_latency_ms = (eval_duration / len(test_texts)) * 1000

    print("\n" + "=" * 50, flush=True)
    print("           MODEL TEST EVALUATION RESULTS          ", flush=True)
    print("=" * 50, flush=True)
    print(f"Overall Accuracy:  {acc * 100:.2f}%", flush=True)
    print(f"Macro F1-Score:    {macro_f1:.4f}", flush=True)
    print(f"Avg Inference Latency: {avg_latency_ms:.3f} ms/sample", flush=True)
    print(f"Fallback Rate (<0.60 conf / sarcasm): {sum(fallback_flags) / len(fallback_flags) * 100:.1f}%", flush=True)
    print("-" * 50, flush=True)
    print("\nClassification Report:", flush=True)
    report = classification_report(test_labels, pred_labels, digits=4)
    print(report, flush=True)

    print("\nConfusion Matrix:", flush=True)
    labels_order = sorted(list(set(test_labels)))
    cm = confusion_matrix(test_labels, pred_labels, labels=labels_order)
    print(f"Labels: {labels_order}", flush=True)
    print(cm, flush=True)

    # Save model
    model.save(model_output_path)
    file_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
    print(f"\nTrained model successfully serialized to: {model_output_path}", flush=True)
    print(f"Compressed Model Disk Size: {file_size_mb:.2f} MB", flush=True)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency_ms,
        "model_path": model_output_path,
        "file_size_mb": file_size_mb
    }

if __name__ == "__main__":
    run_training()
