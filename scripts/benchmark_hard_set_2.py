import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.serving.app import app

SET_2_EXAMPLES = [
    {"id": 1, "text": "Finally something in this country works on time. My cremation is scheduled for 3pm sharp", "true": "Negative", "acceptable": ["negative"]},
    {"id": 2, "text": "They said the update would fix the bug. They were technically correct", "true": "Negative", "acceptable": ["negative"]},
    {"id": 3, "text": "This is the content Elon warned us about", "true": "Neutral/Negative", "acceptable": ["neutral", "negative"]},
    {"id": 4, "text": "I trust this company about as much as I trust free wifi at an airport", "true": "Negative", "acceptable": ["negative"]},
    {"id": 5, "text": "Sure, \"10 minute delivery.\" Sure. Jan", "true": "Negative", "acceptable": ["negative"]},
    {"id": 6, "text": "New phone who dis, said no one to this brand ever again", "true": "Negative", "acceptable": ["negative"]},
    {"id": 7, "text": "I'd explain why this is bad but I don't have that kind of time or patience", "true": "Negative", "acceptable": ["negative"]},
    {"id": 8, "text": "This app has more bugs than my grandma's garden and somehow I still use it daily", "true": "Mixed", "acceptable": ["mixed", "negative"]},
    {"id": 9, "text": "The reviews said 5 stars. The reviews lied", "true": "Negative", "acceptable": ["negative"]},
    {"id": 10, "text": "At this point it's basically modern art, nobody knows what it's supposed to do", "true": "Negative", "acceptable": ["negative"]},
    {"id": 11, "text": "I've had worse days. This wasn't one of the worse ones. It also wasn't a good one", "true": "Neutral", "acceptable": ["neutral"]},
    {"id": 12, "text": "Bro invented a new kind of pain and called it \"customer support\"", "true": "Negative", "acceptable": ["negative"]},
    {"id": 13, "text": "It's giving \"we ran out of budget halfway through\" and I respect the hustle", "true": "Mixed", "acceptable": ["mixed", "positive"]},
    {"id": 14, "text": "Room temperature IQ decisions being made in this boardroom, love to see it", "true": "Negative", "acceptable": ["negative"]},
    {"id": 15, "text": "This isn't even my final form of disappointment", "true": "Negative", "acceptable": ["negative"]},
    {"id": 16, "text": "The CEO really said \"let them eat innovation\"", "true": "Negative", "acceptable": ["negative"]},
    {"id": 17, "text": "Ok but why did that hit different though", "true": "Positive", "acceptable": ["positive"]},
    {"id": 18, "text": "This product really understood the assignment (derogatory)", "true": "Negative", "acceptable": ["negative"]},
    {"id": 19, "text": "Not the plot twist I asked for but the plot twist I deserved", "true": "Mixed", "acceptable": ["mixed", "neutral"]},
    {"id": 20, "text": "Every time I think it can't get worse, it graduates", "true": "Negative", "acceptable": ["negative"]},
    {"id": 21, "text": "POV: you're the only one who thinks this is fine", "true": "Negative", "acceptable": ["negative"]},
    {"id": 22, "text": "This company's customer service manual must just say \"vibes only\"", "true": "Negative", "acceptable": ["negative"]},
    {"id": 23, "text": "I can't decide if this is a 2/10 or a 9/10, there is no in-between for me", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 24, "text": "The bar was already on the floor and somehow they added a shovel", "true": "Negative", "acceptable": ["negative"]},
    {"id": 25, "text": "Iconic behavior from a company that peaked in 2015", "true": "Negative", "acceptable": ["negative"]},
    {"id": 26, "text": "This is what happens when nepotism meets a keyboard", "true": "Negative", "acceptable": ["negative"]},
    {"id": 27, "text": "Not to start drama but the drama started itself when I opened the app", "true": "Negative", "acceptable": ["negative"]},
    {"id": 28, "text": "This is objectively terrible and I have never had more fun in my life", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 29, "text": "The support ticket has been \"in progress\" since the Ice Age", "true": "Negative", "acceptable": ["negative"]},
    {"id": 30, "text": "They fixed the bug by making three new ones, efficient", "true": "Negative", "acceptable": ["negative"]},
    {"id": 31, "text": "I have never related to a broken vending machine more than I do right now", "true": "Negative", "acceptable": ["negative"]},
    {"id": 32, "text": "This is fine dining if fine means my card got declined", "true": "Negative", "acceptable": ["negative"]},
    {"id": 33, "text": "We stan a consistently mediocre king", "true": "Mixed", "acceptable": ["mixed", "positive", "negative"]},
    {"id": 34, "text": "This experience really said \"buffering\" the entire time", "true": "Negative", "acceptable": ["negative"]},
    {"id": 35, "text": "I would leave a review but the app crashed before I could, poetic", "true": "Negative", "acceptable": ["negative"]},
    {"id": 36, "text": "Not everyone was blessed with attention to detail apparently", "true": "Negative", "acceptable": ["negative"]},
    {"id": 37, "text": "This company runs on hopes, dreams, and duct tape", "true": "Mixed/Negative", "acceptable": ["mixed", "negative"]},
    {"id": 38, "text": "Ten out of ten, would not do again, five stars", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 39, "text": "The waiting time built character. Mostly resentment, but character", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 40, "text": "This is the “trust the process” of customer service, except the process is broken", "true": "Negative", "acceptable": ["negative"]},
    {"id": 41, "text": "Groundbreaking. They made the bug worse and called it a feature", "true": "Negative", "acceptable": ["negative"]},
    {"id": 42, "text": "I don't know whether to laugh or file a complaint", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 43, "text": "The app really said \"accessibility is a suggestion, not a rule\"", "true": "Negative", "acceptable": ["negative"]},
    {"id": 44, "text": "This is the calm before absolutely nothing happens, as usual", "true": "Negative", "acceptable": ["negative"]},
    {"id": 45, "text": "Every update is a surprise mystery box, sometimes it's features, usually it's bugs", "true": "Mixed", "acceptable": ["mixed", "negative"]},
    {"id": 46, "text": "This company's roadmap is more theoretical than my physics homework", "true": "Negative", "acceptable": ["negative"]},
    {"id": 47, "text": "Certified hood classic of app crashes", "true": "Positive/Neutral", "acceptable": ["positive", "neutral", "mixed"]},
    {"id": 48, "text": "This isn't a bug, it's an unannounced feature reveal", "true": "Negative", "acceptable": ["negative"]},
    {"id": 49, "text": "I've made peace with mediocrity, we coexist now", "true": "Neutral", "acceptable": ["neutral"]},
    {"id": 50, "text": "The silence from support has been deafening, in a refreshing way", "true": "Mixed", "acceptable": ["mixed"]}
]

def run_test_set_2():
    with TestClient(app) as client:
        texts = [ex["text"] for ex in SET_2_EXAMPLES]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"texts": texts, "extract_aspects": True, "enable_fallback": False})
        batch_latency = (time.perf_counter() - t0) * 1000
        
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        
        passed = 0
        failed = []
        
        print("=" * 95)
        print("           HARD SENTIMENT TEST SET 2 (50 'WHY LLMs STRUGGLE' EXAMPLES)           ")
        print("=" * 95)
        
        for i, ex in enumerate(SET_2_EXAMPLES):
            res = results[i]
            pred_label = res["label"].lower()
            is_pass = pred_label in ex["acceptable"]
            
            if is_pass:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                failed.append({
                    "id": ex["id"],
                    "text": ex["text"],
                    "true": ex["true"],
                    "pred": pred_label,
                    "confidence": res["confidence"],
                    "probabilities": res["probabilities"],
                    "reason": res.get("reason")
                })
                
            print(f"#{ex['id']:02d} | True: {ex['true']:<16} | Pred: {pred_label.upper():<8} ({res['confidence']*100:4.1f}%) | {status} | {ex['text'][:55]}...")
            
        print("=" * 95)
        print(f"RESULTS: {passed} / {len(SET_2_EXAMPLES)} PASSED ({passed / len(SET_2_EXAMPLES) * 100:.1f}%)")
        print(f"Batch Latency: {batch_latency:.2f} ms ({batch_latency / len(SET_2_EXAMPLES):.3f} ms / post)")
        print("=" * 95)
        
        if failed:
            print(f"\nFAILED EXAMPLES ({len(failed)}):")
            for f in failed:
                print(f"\n[#{f['id']}] Text: \"{f['text']}\"")
                print(f"       Expected: {f['true']} | Predicted: {f['pred'].upper()} (Conf: {f['confidence']*100:.1f}%)")
                print(f"       Probs: {f['probabilities']} | Reason: {f['reason']}")

if __name__ == "__main__":
    run_test_set_2()
