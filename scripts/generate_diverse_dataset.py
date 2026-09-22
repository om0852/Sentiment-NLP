import os
import sys
import json
import random
from typing import List, Tuple, Dict

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from src.preprocessing.pipeline import PreprocessingPipeline

# 1. TEMPLATES AND VOCABULARY ACROSS DIVERSE DOMAINS

SLANG_POS_SUBJECTS = ["this new update", "the UI", "their workflow", "this feature", "the app", "this tool", "the design", "the backend speed", "their customer service", "the camera mode"]
SLANG_POS_PREDICATES = [
    "is absolute fire no cap 🔥",
    "is genuinely goated fr",
    "slaps so hard honestly",
    "is top tier, easily a massive W",
    "is bussing, zero lag whatsoever",
    "ate and left no crumbs ✨",
    "is an absolute masterpiece 10/10",
    "is giving immaculate vibes",
    "is super clean and valid",
    "is god tier, nobody does it better 👑",
    "has insane rizz, loving every second",
    "is totally based and wholesome",
    "made my day, dying of laughter 💀🤣",
    "is so smooth and buttery"
]

SLANG_NEG_PREDICATES = [
    "is totally mid tbh",
    "is completely cooked, nothing works",
    "is another massive L for the company 🤡",
    "is pure dog water garbage 🗑️",
    "is so cringe and broken 🤮",
    "is an absolute flop of a release",
    "is super sus, feels like a complete scam 👎",
    "got bricked after the latest patch",
    "fell off so hard, used to be good",
    "is living in founders' heads rent free while broken",
    "is giving me a huge ick",
    "feels like a complete ripoff",
    "is full of annoying bugs and crashes 😡"
]

CONTRASTIVE_TEMPLATES = [
    # Positive overall
    ("The onboarding was confusing, but the core engine is so fire and works like a charm! 🔥", "positive"),
    ("Had a small bug yesterday, but support fixed it immediately, huge W! 🙌", "positive"),
    ("A bit pricey honestly, but the productivity boost is top tier and totally worth it.", "positive"),
    ("It took a minute to learn, but now I can't imagine working without it, absolute gem! ✨", "positive"),
    ("Interface looks simple, but the speed under the hood is ridiculously goated.", "positive"),
    ("Setup was slightly annoying, but the results are clean and immaculate.", "positive"),

    # Negative overall
    ("The marketing video looked cool, but the actual software is completely cooked and unusable.", "negative"),
    ("Great color scheme, but everything crashes every two minutes, total garbage 🗑️", "negative"),
    ("They promised 24/7 support, but nobody answered and my account got locked, huge L 🤡", "negative"),
    ("The design is sleek, but the pricing is an absolute ripoff with zero useful features.", "negative"),
    ("Nice idea in theory, but execution is totally mid and plagued with glitches.", "negative"),
    ("It used to be decent, but the new pricing tier ruined everything, avoid at all costs.", "negative"),
]

NEGATION_TEMPLATES = [
    # Negative negations
    ("This is not good at all, extremely disappointed with the build.", "negative"),
    ("Can't recommend this service to anyone, wasted 3 hours.", "negative"),
    ("Won't be renewing my subscription, total headache.", "negative"),
    ("Never seen an app freeze this much, hardly usable.", "negative"),
    ("It is not even remotely close to the advertised quality.", "negative"),
    ("Don't waste your money here, terrible experience.", "negative"),
    ("Couldn't get a single export to finish without crashing.", "negative"),
    ("There is no excuse for this level of downtime.", "negative"),

    # Positive negations (negation of negative words = positive)
    ("Not bad at all, actually pleasantly surprised with the speed! ✨", "positive"),
    ("Can't complain, runs without any lag on my laptop 💯", "positive"),
    ("Never had a single crash in 6 months of daily use, love it! ❤️", "positive"),
    ("Won't switch to any other tool, this one is the best by far.", "positive"),
    ("Not going to lie, this exceeded all my expectations 🔥", "positive"),
    ("No regrets buying the pro plan, paid for itself in a week.", "positive"),
    ("Hardly felt any learning curve, super intuitive and clean.", "positive")
]

SARCASM_TEMPLATES = [
    ("Oh wow, what a genius move breaking production on a Friday afternoon /s", "negative"),
    ("Totally love waiting 45 minutes on hold just to get disconnected, great job /s", "negative"),
    ("Yeah right, another monthly subscription, exactly what humanity needed 🙄", "negative"),
    ("Big surprise, the server crashed again right before the demo /s", "negative"),
    ("Totally normal for an app to use 8GB of RAM to render a button /s 🤡", "negative"),
    ("Such a wonderful update, now none of my saved files open anymore /s", "negative"),
    ("Oh brilliant, double charged my card and deleted my account, masterclass in tech /s", "negative")
]

NEUTRAL_DOMAINS = [
    # Tech & Infrastructure
    "The scheduled database migration will occur on Saturday at 2:00 AM UTC.",
    "Release version 3.2.0 contains security patches for CVE-2026-1182.",
    "Please review the API rate limit specifications in section 4 of the documentation.",
    "The webhook endpoint received a 200 OK status code from the gateway.",
    "System latency averaged 42ms across all regional clusters yesterday.",
    "We have published the updated architectural diagram for the microservices layer.",
    "How do I configure the reverse proxy with Nginx for WebSocket connections?",
    "The CPU utilization remained steady at 38% during the reporting window.",

    # Business & Workplace
    "Quarterly financial statements have been uploaded to the investor relations portal.",
    "The weekly sprint planning call has been rescheduled to Thursday 11 AM.",
    "Meeting minutes from yesterday's stakeholder sync are attached to this email.",
    "Please submit your travel expense reports before the end of the month.",
    "The office will be closed next Monday in observance of the public holiday.",
    "Here is the summary of attendee demographics from the annual conference.",
    "The human resources department announced the upcoming open enrollment dates.",

    # Social & General Inquiries
    "What time does the keynote session begin tomorrow morning?",
    "Can anyone confirm if the shuttle bus runs on Sundays?",
    "Here is the link to the live stream for anyone who cannot attend in person.",
    "The library will extend its operating hours during finals week.",
    "Looking for recommendations on lightweight Python libraries for data processing.",
    "The temperature in Chicago is expected to reach 65 degrees today.",
    "Flight AA142 has arrived at gate B12 on schedule."
]

def generate_synthetic_samples(count: int = 6000) -> List[Tuple[str, str]]:
    """Generates a rich, diverse set of realistic social samples."""
    samples = []
    
    # 1. Contrastive & complex sentences
    for _ in range(count // 6):
        s, label = random.choice(CONTRASTIVE_TEMPLATES)
        samples.append((s, label))
        
    # 2. Negation challenges
    for _ in range(count // 6):
        s, label = random.choice(NEGATION_TEMPLATES)
        samples.append((s, label))
        
    # 3. Sarcasm-lite
    for _ in range(count // 10):
        s, label = random.choice(SARCASM_TEMPLATES)
        samples.append((s, label))
        
    # 4. Slang generative combinations (Positive)
    for _ in range(count // 4):
        subj = random.choice(SLANG_POS_SUBJECTS)
        pred = random.choice(SLANG_POS_PREDICATES)
        prefix = random.choice(["", "Honestly, ", "Bro, ", "Ngl, ", "Tbh, ", "Yo, "])
        text = f"{prefix}{subj} {pred}".strip()
        samples.append((text, "positive"))
        
    # 5. Slang generative combinations (Negative)
    for _ in range(count // 4):
        subj = random.choice(SLANG_POS_SUBJECTS)
        pred = random.choice(SLANG_NEG_PREDICATES)
        prefix = random.choice(["", "Honestly, ", "Bro, ", "Ngl, ", "Tbh, ", "Yo, "])
        text = f"{prefix}{subj} {pred}".strip()
        samples.append((text, "negative"))
        
    # 6. Neutrals
    for _ in range(count // 4):
        base_neutral = random.choice(NEUTRAL_DOMAINS)
        samples.append((base_neutral, "neutral"))
        
    return samples

def augment_and_save_dataset():
    data_dir = os.path.join(project_root, "data", "processed")
    pipeline = PreprocessingPipeline()
    
    print("Loading existing verified datasets...", flush=True)
    with open(os.path.join(data_dir, "train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, "val.json"), "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(os.path.join(data_dir, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print(f"Existing counts: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}", flush=True)
    
    # Generate 6,000 diverse samples
    print("Synthesizing 6,000 diverse samples across slang, emojis, negations, contrastive clauses, and domains...", flush=True)
    raw_synth = generate_synthetic_samples(count=6000)
    random.seed(42)
    random.shuffle(raw_synth)
    
    processed_synth = []
    for raw_text, label in raw_synth:
        proc_text, _ = pipeline.process(raw_text)
        processed_synth.append({
            "raw_text": raw_text,
            "processed_text": proc_text,
            "label": label,
            "source": "diverse_synthetic_v2"
        })
        
    # Split synthetic data 70/15/15
    n_synth = len(processed_synth)
    n_train = int(n_synth * 0.70)
    n_val = int(n_synth * 0.15)
    
    synth_train = processed_synth[:n_train]
    synth_val = processed_synth[n_train:n_train + n_val]
    synth_test = processed_synth[n_train + n_val:]
    
    # Merge
    augmented_train = train_data + synth_train
    augmented_val = val_data + synth_val
    augmented_test = test_data + synth_test
    
    random.shuffle(augmented_train)
    random.shuffle(augmented_val)
    random.shuffle(augmented_test)
    
    # Save back
    with open(os.path.join(data_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(augmented_train, f, indent=2)
    with open(os.path.join(data_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(augmented_val, f, indent=2)
    with open(os.path.join(data_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(augmented_test, f, indent=2)
        
    print(f"\n--- Augmented Dataset Summary ---", flush=True)
    print(f"Total Samples: {len(augmented_train) + len(augmented_val) + len(augmented_test)}", flush=True)
    print(f"Augmented Train: {len(augmented_train)}", flush=True)
    print(f"Augmented Val:   {len(augmented_val)}", flush=True)
    print(f"Augmented Test:  {len(augmented_test)}", flush=True)

if __name__ == "__main__":
    augment_and_save_dataset()
