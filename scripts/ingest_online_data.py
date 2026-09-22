import os
import sys
import json
import random
from typing import List, Dict

# Ensure console supports UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
sys.path.insert(0, project_root)

from src.preprocessing.pipeline import PreprocessingPipeline

LABEL_MAP = {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
}

def load_tweeteval_split(split_name: str, raw_dir: str):
    text_file = os.path.join(raw_dir, f"{split_name}_text.txt")
    label_file = os.path.join(raw_dir, f"{split_name}_labels.txt")
    
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f]
        
    with open(label_file, "r", encoding="utf-8") as f:
        num_labels = [line.strip() for line in f]
        
    return texts, [LABEL_MAP[l] for l in num_labels]

def run_ingest():
    raw_dir = os.path.join(project_root, "data", "raw", "tweeteval")
    proc_dir = os.path.join(project_root, "data", "processed")
    pipeline = PreprocessingPipeline()
    
    print("Loading existing domain & MongoDB verified datasets...", flush=True)
    with open(os.path.join(proc_dir, "train.json"), "r", encoding="utf-8") as f:
        existing_train = json.load(f)
    with open(os.path.join(proc_dir, "val.json"), "r", encoding="utf-8") as f:
        existing_val = json.load(f)
    with open(os.path.join(proc_dir, "test.json"), "r", encoding="utf-8") as f:
        existing_test = json.load(f)
        
    print(f"Existing Dataset: Train={len(existing_train)}, Val={len(existing_val)}, Test={len(existing_test)}", flush=True)
    
    # 1. Load TweetEval Train (45,615 tweets)
    print("Loading TweetEval training set...", flush=True)
    train_texts, train_labels = load_tweeteval_split("train", raw_dir)
    print(f"Loaded {len(train_texts):,} online training tweets.", flush=True)
    
    # Balance classes from TweetEval (e.g. up to 10,000 per class = 30,000 online tweets)
    online_by_class: Dict[str, List[Dict]] = {"negative": [], "neutral": [], "positive": []}
    target_per_class = 10000
    
    print("Preprocessing online tweets through linguistic pipeline...", flush=True)
    for i, (text, label) in enumerate(zip(train_texts, train_labels)):
        if len(online_by_class[label]) < target_per_class:
            proc_text, _ = pipeline.process(text)
            online_by_class[label].append({
                "raw_text": text,
                "processed_text": proc_text,
                "label": label,
                "source": "tweeteval_online"
            })
            
    online_samples = []
    for cls, items in online_by_class.items():
        print(f"  Sampled {cls.capitalize()}: {len(items):,} online tweets", flush=True)
        online_samples.extend(items)
        
    random.seed(42)
    random.shuffle(online_samples)
    
    # 70% train / 15% val / 15% test of the online samples
    n_online = len(online_samples)
    n_tr = int(n_online * 0.70)
    n_va = int(n_online * 0.15)
    
    online_train = online_samples[:n_tr]
    online_val = online_samples[n_tr:n_tr + n_va]
    online_test = online_samples[n_tr + n_va:]
    
    # Combine with domain data (keep domain gold slang well-represented)
    unified_train = existing_train + online_train
    unified_val = existing_val + online_val
    unified_test = existing_test + online_test
    
    random.shuffle(unified_train)
    random.shuffle(unified_val)
    random.shuffle(unified_test)
    
    print("\n--- Unified Multi-Source Dataset Created ---", flush=True)
    print(f"Total Unified Samples: {len(unified_train) + len(unified_val) + len(unified_test):,}")
    print(f"Train Set: {len(unified_train):,} ({len(existing_train):,} domain + {len(online_train):,} online)")
    print(f"Val Set:   {len(unified_val):,} ({len(existing_val):,} domain + {len(online_val):,} online)")
    print(f"Test Set:  {len(unified_test):,} ({len(existing_test):,} domain + {len(online_test):,} online)")
    
    with open(os.path.join(proc_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(unified_train, f, indent=2)
    with open(os.path.join(proc_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(unified_val, f, indent=2)
    with open(os.path.join(proc_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(unified_test, f, indent=2)
        
    print("\nSuccessfully updated processed datasets!", flush=True)

if __name__ == "__main__":
    run_ingest()
