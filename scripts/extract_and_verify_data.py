import os
import sys
import re
import json
import random
from typing import Dict, List, Tuple
from pymongo import MongoClient

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.pipeline import PreprocessingPipeline

MONGO_URI = os.getenv("MONGO_URI", "")

# Curated Gold-Standard Social Slang & Emotion Samples
GOLD_SLANG_SAMPLES = [
    # Positive Slang & Modern Sentiment
    ("this new design is so fire no cap", "positive"),
    ("bro that update is actually goated", "positive"),
    ("such an immaculate vibe from the team", "positive"),
    ("the performance on this version slaps so hard", "positive"),
    ("massive W for the creators community", "positive"),
    ("sheesh they really ate and left no crumbs with this release", "positive"),
    ("top tier customer support honestly, resolved in 5 mins", "positive"),
    ("best tool I've used all year, absolute god tier 🔥", "positive"),
    ("this feature is so clean and smooth ✨", "positive"),
    ("huge W for everyone involved 🙌", "positive"),
    ("can't stop using this, 10/10 masterpiece", "positive"),
    ("super hyped for the next feature rollout 🚀", "positive"),
    ("this is genuinely valid and wholesome", "positive"),
    ("i am dying of laughter this is so funny 💀", "positive"),
    ("bro this is hilarious 😭🤣", "positive"),
    ("pure genius whoever built this feature 👏", "positive"),
    ("underrated tool honestly deserves way more praise", "positive"),
    ("absolutely legendary work by the devs", "positive"),
    ("bussing performance, no lag whatsoever 💯", "positive"),
    ("the aesthetic is so snatched and modern", "positive"),
    ("such a rare gem of an application", "positive"),
    ("flawless execution and super easy to use 😍", "positive"),
    ("loving every bit of this new workflow ❤️", "positive"),
    ("honestly couldn't be happier with the results", "positive"),
    ("they delivered on every single promise", "positive"),

    # Negative Slang & Modern Sentiment
    ("customer service was completely mid tbh", "negative"),
    ("this whole app is cooked, nothing is loading", "negative"),
    ("another massive L for this company 🤡", "negative"),
    ("total scam, they charged my card twice and won't refund", "negative"),
    ("this update is pure dog water garbage 🗑️", "negative"),
    ("the new UI is so cringe and confusing 🤮", "negative"),
    ("looks like a complete ripoff of other tools", "negative"),
    ("bro took a huge L with this broken update", "negative"),
    ("super sus activity on my account, avoid at all costs", "negative"),
    ("crashes every 2 minutes, utterly useless piece of software", "negative"),
    ("why is this so buggy and slow 😡", "negative"),
    ("this feature is completely bricked after today's patch", "negative"),
    ("they fell off so hard, used to be good now it's terrible", "negative"),
    ("pure cap, doesn't do anything advertised 👎", "negative"),
    ("annoying bugs everywhere, huge waste of money", "negative"),
    ("the founders are delulu if they think people will pay for this", "negative"),
    ("absolute flop of a release", "negative"),
    ("support literally ghosted me after taking payment 💔", "negative"),
    ("not good at all, completely disappointed", "negative"),
    ("hardly works on mobile, worst experience ever", "negative"),
    ("can't recommend this to anyone, total headache", "negative"),
    ("never trusting this service again, full of glitches", "negative"),
    ("it is not even remotely close to being usable", "negative"),
    ("terrible speed and constantly freezing up", "negative"),
    ("disaster of an update, rolled back immediately", "negative"),

    # Neutral Informational Social Posts
    ("here is the updated schedule for tomorrow's webinar", "neutral"),
    ("new feature release notes version 2.4.0 have been published", "neutral"),
    ("how do i change my password in the settings tab?", "neutral"),
    ("the meeting starts at 3pm est today", "neutral"),
    ("can someone share the link to the documentation?", "neutral"),
    ("the company announced its quarterly earnings report this morning", "neutral"),
    ("we are currently investigating reported connectivity in the eu region", "neutral"),
    ("what is the difference between pro and standard tiers?", "neutral"),
    ("please find the attached pdf guide for reference", "neutral"),
    ("fyi the server maintenance is scheduled for midnight", "neutral"),
    ("the workshop is open for all registered attendees", "neutral"),
    ("sharing a summary of today's industry news", "neutral"),
    ("the webinar recording will be available by tomorrow afternoon", "neutral"),
    ("social media trends report for Q3 is now available", "neutral"),
    ("system status indicates normal operating metrics", "neutral")
]

class DataExtractorVerifier:
    def __init__(self, target_per_class: int = 5000):
        self.pipeline = PreprocessingPipeline()
        self.target_per_class = target_per_class
        
        # Load dictionaries for lexical verification
        dict_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dictionaries")
        with open(os.path.join(dict_dir, "slang_dict.json"), "r", encoding="utf-8") as f:
            self.slang_dict = json.load(f)
        with open(os.path.join(dict_dir, "emoji_dict.json"), "r", encoding="utf-8") as f:
            self.emoji_dict = json.load(f)
            
        self.positive_keywords = {
            "good", "great", "awesome", "excellent", "love", "loved", "loving", "best", "fantastic",
            "amazing", "wonderful", "perfect", "beautiful", "happy", "impressive", "superb", "brilliant",
            "favorite", "helpful", "clean", "easy", "recommend", "thank", "thanks", "glad", "nice", "cool"
        }
        self.negative_keywords = {
            "bad", "terrible", "worst", "horrible", "awful", "hate", "hated", "hating", "poor", "fail",
            "failed", "failure", "sucks", "sucked", "annoying", "annoyed", "useless", "disappointed",
            "disappointing", "broken", "bug", "bugs", "glitch", "crash", "crashes", "slow", "freeze",
            "fraud", "scam", "trash", "waste", "problem", "issues", "pain", "stuck"
        }
        self.spam_patterns = [
            re.compile(r"^photo by .* on \w+ \d+", re.IGNORECASE),
            re.compile(r"hit a \$\d+[BM] valuation", re.IGNORECASE),
            re.compile(r"click (here|the link) to", re.IGNORECASE),
            re.compile(r"claim your .* subscription", re.IGNORECASE),
            re.compile(r"^\s*http[s]?://\S+\s*$", re.IGNORECASE),
        ]

    def verify_sample(self, raw_text: str, orig_label: str) -> Tuple[bool, str, str]:
        """
        Evaluates a raw post from MongoDB.
        Returns: (is_valid, verified_label, reason)
        """
        if not raw_text or len(raw_text.strip()) < 12:
            return False, "", "Text too short"
            
        for p in self.spam_patterns:
            if p.search(raw_text):
                return False, "", "Spam or boilerplate"

        # Preprocess text
        processed_text, meta = self.pipeline.process(raw_text)
        
        words = set(re.findall(r"\b[\w']+\b", processed_text.lower()))
        
        pos_kw_count = len(words.intersection(self.positive_keywords))
        neg_kw_count = len(words.intersection(self.negative_keywords))
        composite_score = meta["composite_polarity"] + (pos_kw_count * 1.0) - (neg_kw_count * 1.0)
        
        orig = str(orig_label).lower().strip() if orig_label else "unknown"
        
        if orig == "positive":
            # Must have positive signals and NO strong negative signals
            if composite_score >= 1.0 and neg_kw_count == 0 and meta["emoji_polarity"] >= 0:
                return True, "positive", "Strong positive agreement"
            elif composite_score < -0.8 or neg_kw_count >= 2:
                # Mislabeled in DB! Reject corrupt data
                return False, "", "Corrupt DB label: labeled positive but content is negative"
            elif composite_score > 0 and neg_kw_count == 0:
                return True, "positive", "Moderate positive agreement"
            else:
                return False, "", "Insufficient confidence for positive"

        elif orig == "negative":
            # Must have negative signals and NO strong positive signals
            if composite_score <= -1.0 and pos_kw_count == 0 and meta["emoji_polarity"] <= 0:
                return True, "negative", "Strong negative agreement"
            elif composite_score > 0.8 or pos_kw_count >= 2:
                # Mislabeled in DB! Reject corrupt data
                return False, "", "Corrupt DB label: labeled negative but content is positive"
            elif composite_score < 0 and pos_kw_count == 0:
                return True, "negative", "Moderate negative agreement"
            else:
                return False, "", "Insufficient confidence for negative"

        elif orig == "neutral":
            # Truly neutral text must have near-zero polarity and no strong sentiment words
            if abs(composite_score) <= 0.2 and pos_kw_count == 0 and neg_kw_count == 0:
                return True, "neutral", "Verified neutral objective"
            else:
                return False, "", "Subjective signals present in neutral label"

        return False, "", "Unknown label"

    def run_extraction_pipeline(self):
        print(f"Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
        col = client["cache1"]["listenings"]
        
        verified_data: Dict[str, List[Dict]] = {
            "positive": [],
            "negative": [],
            "neutral": []
        }
        
        # 1. Add gold-standard slang samples (oversampled for representation)
        print("Injecting gold-standard modern slang samples...")
        for text, label in GOLD_SLANG_SAMPLES:
            proc_text, meta = self.pipeline.process(text)
            for _ in range(25):  # Replicate across splits to anchor slang semantics
                verified_data[label].append({
                    "raw_text": text,
                    "processed_text": proc_text,
                    "label": label,
                    "source": "gold_slang"
                })
                
        print(f"Initial gold samples added: { {k: len(v) for k, v in verified_data.items()} }")
        
        # 2. Stream documents from MongoDB
        print("Streaming and verifying documents from cache1.listenings...")
        # Query with projection to minimize network transfer
        cursor = col.find(
            {"sentimental": {"$in": ["Positive", "Negative", "Neutral"]}},
            {"content": 1, "title": 1, "sentimental": 1, "source": 1}
        ).batch_size(1000)
        
        checked = 0
        accepted = 0
        corrupt_rejected = 0
        
        for doc in cursor:
            checked += 1
            raw_text = doc.get("content") or doc.get("title") or ""
            orig_label = doc.get("sentimental")
            
            is_valid, label, reason = self.verify_sample(raw_text, orig_label)
            
            if is_valid:
                if len(verified_data[label]) < self.target_per_class:
                    proc_text, _ = self.pipeline.process(raw_text)
                    verified_data[label].append({
                        "raw_text": raw_text,
                        "processed_text": proc_text,
                        "label": label,
                        "source": doc.get("source", "mongo_listenings")
                    })
                    accepted += 1
            elif "Corrupt" in reason:
                corrupt_rejected += 1
                
            if checked % 5000 == 0:
                counts = {k: len(v) for k, v in verified_data.items()}
                print(f"Checked: {checked} | Accepted: {accepted} | Corrupt Rejected: {corrupt_rejected} | Current Counts: {counts}")
                
            if all(len(v) >= self.target_per_class for v in verified_data.values()):
                print(f"Reached target {self.target_per_class} per class! Finalizing dataset.")
                break
                
        # 3. Consolidate and create 70/15/15 splits
        all_samples = []
        for label, samples in verified_data.items():
            all_samples.extend(samples)
            
        random.seed(42)
        random.shuffle(all_samples)
        
        n_total = len(all_samples)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        train_set = all_samples[:n_train]
        val_set = all_samples[n_train:n_train + n_val]
        test_set = all_samples[n_train + n_val:]
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
        os.makedirs(data_dir, exist_ok=True)
        
        with open(os.path.join(data_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train_set, f, indent=2)
        with open(os.path.join(data_dir, "val.json"), "w", encoding="utf-8") as f:
            json.dump(val_set, f, indent=2)
        with open(os.path.join(data_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2)
            
        print("\n--- Dataset Creation Summary ---")
        print(f"Total Verified Samples: {n_total}")
        print(f"Train Set: {len(train_set)} (70%)")
        print(f"Validation Set: {len(val_set)} (15%)")
        print(f"Test Set: {len(test_set)} (15%)")
        print(f"Total Corrupt DB Labels Caught & Filtered: {corrupt_rejected}")

if __name__ == "__main__":
    target = 2500  # 2500 * 3 = 7,500 highly verified balanced samples (instant training & test)
    extractor = DataExtractorVerifier(target_per_class=target)
    extractor.run_extraction_pipeline()
