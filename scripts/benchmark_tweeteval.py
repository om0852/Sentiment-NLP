import os
import sys
import time
import httpx
import json
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure console supports UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root in python path
project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
sys.path.insert(0, project_root)

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.models.tfidf_classifier import TfidfSentimentClassifier

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def fetch_tweeteval_test_set() -> List[Dict[str, str]]:
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    cache_file = os.path.join(raw_dir, "tweeteval_sentiment_test.json")

    if os.path.exists(cache_file):
        print(f"Loading cached TweetEval test set from: {cache_file}", flush=True)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Downloading official TweetEval sentiment test set (~12,284 tweets)...", flush=True)
    text_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_text"
    label_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels"

    with httpx.Client(timeout=30.0) as client:
        r_text = client.get(text_url)
        r_label = client.get(label_url)

    if r_text.status_code != 200 or r_label.status_code != 200:
        raise RuntimeError(f"Failed to fetch TweetEval test set. Status: text={r_text.status_code}, label={r_label.status_code}")

    texts = [t.strip() for t in r_text.text.strip().split("\n")]
    labels = [LABEL_MAP[int(l.strip())] for l in r_label.text.strip().split("\n")]

    dataset = [{"text": t, "label": l} for t, l in zip(texts, labels)]
    print(f"Successfully downloaded {len(dataset):,} tweets!", flush=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset

def run_evaluation():
    dataset = fetch_tweeteval_test_set()
    n_samples = len(dataset)
    print(f"Evaluating model on all {n_samples:,} official TweetEval benchmark tweets...", flush=True)

    # Class distribution in test set
    label_counts = {}
    for d in dataset:
        l = d["label"]
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"Test Set Ground Truth Distribution: {label_counts}", flush=True)

    # Load engine
    pipeline = PreprocessingPipeline()
    context_analyzer = ContextAnalyzer()
    model = TfidfSentimentClassifier()
    model_path = os.path.join(project_root, "models", "sentiment_model.joblib")
    model.load(model_path)

    # Run batch inference
    raw_texts = [d["text"] for d in dataset]
    true_labels = [d["label"] for d in dataset]

    print("Running batch preprocessing & inference...", flush=True)
    t0 = time.perf_counter()

    batch_size = 500
    pred_labels = []
    confidences = []

    for i in range(0, n_samples, batch_size):
        batch_raw = raw_texts[i:i + batch_size]
        batch_proc = [pipeline.process(t)[0] for t in batch_raw]
        batch_preds = model.predict_batch(batch_proc)

        for j, p in enumerate(batch_preds):
            final_label, conf, _, _ = context_analyzer.analyze(batch_raw[j], p["label"], p["confidence"])
            # Map 'mixed' to nearest class or keep as is (TweetEval only has pos/neg/neu)
            if final_label == "mixed":
                final_label = p["label"]
            pred_labels.append(final_label)
            confidences.append(conf)

    elapsed_time = time.perf_counter() - t0
    avg_latency_ms = (elapsed_time / n_samples) * 1000
    throughput = n_samples / elapsed_time

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    print("\n" + "=" * 65)
    print("      OFFICIAL TWEETEVAL (SEMEVAL) BENCHMARK TEST RESULTS      ")
    print("=" * 65)
    print(f"Total Test Samples:    {n_samples:,} real-world tweets")
    print(f"Overall Accuracy:      {acc * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1:.4f}")
    print(f"Total Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"Average Latency:       {avg_latency_ms:.3f} ms / tweet")
    print(f"Effective Throughput:  {throughput:,.0f} tweets / second")
    print("-" * 65)
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, digits=4))

    print("Confusion Matrix:")
    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
    print(f"Labels: {labels_order}")
    print(cm)
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
