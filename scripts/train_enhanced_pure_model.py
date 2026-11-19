import os
import sys
import json
import time

# Repo root (for src imports and model output) and data root (for raw/processed data).
# Set SENTIMENT_DATA_ROOT to read data from another checkout (data/ is gitignored).
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", project_root)
sys.path.insert(0, project_root)

from src.models.pure_tfidf import PureSentimentClassifier

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def load_all_training_data():
    texts = []
    labels = []

    # 1. Load TweetEval Train + Val only. The test split is the held-out benchmark
    #    used by scripts/benchmark_tweeteval.py and must never be trained on.
    tweeteval_dir = os.path.join(data_root, "data", "raw", "tweeteval")
    for split in ["train", "val"]:
        t_file = os.path.join(tweeteval_dir, f"{split}_text.txt")
        l_file = os.path.join(tweeteval_dir, f"{split}_labels.txt")
        if os.path.exists(t_file) and os.path.exists(l_file):
            print(f"Loading TweetEval split: {split}...")
            with open(t_file, "r", encoding="utf-8") as f_t, open(l_file, "r", encoding="utf-8") as f_l:
                t_lines = [line.strip() for line in f_t]
                l_lines = [LABEL_MAP[int(line.strip())] for line in f_l]
                texts.extend(t_lines)
                labels.extend(l_lines)

    # 2. Load Processed JSON splits if available
    proc_dir = os.path.join(data_root, "data", "processed")
    for split in ["train", "val", "test"]:
        p_file = os.path.join(proc_dir, f"{split}.json")
        if os.path.exists(p_file):
            print(f"Loading Processed split: {split}.json...")
            with open(p_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    txt = item.get("processed_text") or item.get("raw_text") or item.get("text")
                    lbl = item.get("label", "").lower().strip()
                    if txt and lbl in ["positive", "negative", "neutral"]:
                        texts.append(txt)
                        labels.append(lbl)

    # 3. Drop anything identical to a TweetEval test tweet, in case another source contains it
    test_file = os.path.join(tweeteval_dir, "test_text.txt")
    if os.path.exists(test_file):
        norm = lambda s: " ".join(s.lower().split())
        with open(test_file, "r", encoding="utf-8") as f:
            held_out = {norm(line) for line in f}
        kept = [(t, l) for t, l in zip(texts, labels) if norm(t) not in held_out]
        removed = len(texts) - len(kept)
        if removed:
            print(f"Removed {removed:,} samples that duplicate TweetEval test tweets")
        texts = [t for t, _ in kept]
        labels = [l for _, l in kept]

    print(f"Total Combined Training Samples: {len(texts):,}")
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"Class Distribution: {label_counts}")
    return texts, labels

def main():
    print("=" * 70)
    print("      TRAINING ENHANCED PURE PYTHON SENTIMENT ENGINE MODEL      ")
    print("=" * 70)
    
    texts, labels = load_all_training_data()
    if not texts:
        print("Error: No training data found!")
        return

    # Initialize enhanced model with subword n-grams and 30,000 max features
    model = PureSentimentClassifier(
        max_features=30000,
        lr=0.4,
        epochs=8,
        l2_reg=1e-5
    )

    t0 = time.perf_counter()
    print("\nFitting model with subword character 3-grams and word n-grams...", flush=True)
    model.fit(texts, labels)
    elapsed = time.perf_counter() - t0
    print(f"Training completed successfully in {elapsed:.2f} seconds!", flush=True)

    # Save to models/sentiment_model.json.gz
    output_path = os.path.join(project_root, "models", "sentiment_model.json.gz")
    model.save(output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel saved to: {output_path}")
    print(f"Model Disk Size: {file_size_mb:.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    main()
