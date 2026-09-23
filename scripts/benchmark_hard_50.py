import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.serving.app import app

HARD_50_EXAMPLES = [
    {"id": 1, "text": "Oh great, my flight got delayed by 6 hours AGAIN 🙃 love spending my birthday in an airport lounge eating stale sandwiches, living my best life fr fr 💀", "true": "Negative", "acceptable": ["negative"]},
    {"id": 2, "text": "Not gonna lie, I thought this phone would be mid but ngl it's actually not bad at all, kinda impressed ngl", "true": "Positive", "acceptable": ["positive"]},
    {"id": 3, "text": "The service was painfully slow and the waiter forgot our order twice, but honestly the biryani was so good I'd still come back fr", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 4, "text": "bro really said that to my face and I just 😭😭😭 no cap I was NOT ready for that comment 💀🔥", "true": "Negative", "acceptable": ["negative"]},
    {"id": 5, "text": "yeah that's exactly how I wanted my Monday to go 👍", "true": "Negative", "acceptable": ["negative"]},
    {"id": 6, "text": "This is fine. Everything is fine. Totally fine. 🔥🏠🐶☕", "true": "Negative", "acceptable": ["negative"]},
    {"id": 7, "text": "Can't believe I paid full price for this. Worth every rupee though, no complaints here", "true": "Mixed", "acceptable": ["mixed", "positive"]},
    {"id": 8, "text": "Wow. Just wow. Didn't expect that from you honestly", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed"]},
    {"id": 9, "text": "this app used to be trash but not gonna lie the new update slaps", "true": "Positive", "acceptable": ["positive"]},
    {"id": 10, "text": "I guess it works? if you're into stuff that barely functions lol", "true": "Negative", "acceptable": ["negative"]},
    {"id": 11, "text": "Ngl she ate that performance, no printer could've printed that better 💅", "true": "Positive", "acceptable": ["positive"]},
    {"id": 12, "text": "My package arrived broken, customer support was actually super helpful and fixed it in 10 mins though, can't even be mad", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 13, "text": "This restaurant really said \"let them eat cold food\" 💀", "true": "Negative", "acceptable": ["negative"]},
    {"id": 14, "text": "Not the best, not the worst. It exists.", "true": "Neutral", "acceptable": ["neutral"]},
    {"id": 15, "text": "I'm speechless. Genuinely speechless right now", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed"]},
    {"id": 16, "text": "lowkey this might be the worst best decision I've ever made", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 17, "text": "thanks for absolutely nothing 🙂", "true": "Negative", "acceptable": ["negative"]},
    {"id": 18, "text": "it's giving \"I have no idea what I'm doing\" energy and I'm here for it", "true": "Positive", "acceptable": ["positive"]},
    {"id": 19, "text": "the food was okay, the vibe was immaculate, but I probably won't come back because of the price", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 20, "text": "congrats on being consistently disappointing 🎉", "true": "Negative", "acceptable": ["negative"]},
    {"id": 21, "text": "this update fixed nothing and broke everything, couldn't ask for more 🙃", "true": "Negative", "acceptable": ["negative"]},
    {"id": 22, "text": "I don't hate it", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed", "positive"]},
    {"id": 23, "text": "the movie was mid but the popcorn went hard ngl", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 24, "text": "say less, this is EXACTLY what I needed today 🔥", "true": "Positive", "acceptable": ["positive"]},
    {"id": 25, "text": "oh you're SO funny 🙄", "true": "Negative", "acceptable": ["negative"]},
    {"id": 26, "text": "it's not you, it's the product. actually wait, it might be both", "true": "Negative", "acceptable": ["negative"]},
    {"id": 27, "text": "absolutely unhinged behavior and I respect it so much", "true": "Positive", "acceptable": ["positive"]},
    {"id": 28, "text": "he really woke up and chose violence today 😭", "true": "Neutral/Positive", "acceptable": ["neutral", "positive"]},
    {"id": 29, "text": "I've seen better, I've seen worse. Moving on.", "true": "Neutral", "acceptable": ["neutral"]},
    {"id": 30, "text": "this is the content I signed up for, unfortunately", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 31, "text": "10/10 would NOT recommend, still can't stop thinking about it though", "true": "Mixed", "acceptable": ["mixed", "negative"]},
    {"id": 32, "text": "the audacity of this app to crash during the ONE important call 💀💀", "true": "Negative", "acceptable": ["negative"]},
    {"id": 33, "text": "not to be dramatic but this might genuinely be the best thing that's happened to me all year", "true": "Positive", "acceptable": ["positive"]},
    {"id": 34, "text": "this is peak comedy and I'm not even mad it broke", "true": "Mixed", "acceptable": ["mixed", "positive"]},
    {"id": 35, "text": "wow they really let anyone ship code these days huh", "true": "Negative", "acceptable": ["negative"]},
    {"id": 36, "text": "I'm crying 😭 this is the funniest/saddest thing I've seen today, not sure which", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed"]},
    {"id": 37, "text": "bffr, that ending was actually so good it hurt", "true": "Positive", "acceptable": ["positive"]},
    {"id": 38, "text": "can't lie, kind of a letdown but also kind of iconic", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 39, "text": "this is why we can't have nice things", "true": "Negative", "acceptable": ["negative"]},
    {"id": 40, "text": "oh nooo anyway 💅", "true": "Neutral/Negative", "acceptable": ["neutral", "negative"]},
    {"id": 41, "text": "I was NOT prepared for how good this was ngl", "true": "Positive", "acceptable": ["positive"]},
    {"id": 42, "text": "this company really said \"customer service is dead\" and meant it", "true": "Negative", "acceptable": ["negative"]},
    {"id": 43, "text": "it's giving mixed signals, kind of like this review", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed"]},
    {"id": 44, "text": "ate and left no crumbs, unlike my patience after waiting 2 hours", "true": "Mixed", "acceptable": ["mixed"]},
    {"id": 45, "text": "I have thoughts. Many thoughts. None of them nice", "true": "Negative", "acceptable": ["negative"]},
    {"id": 46, "text": "the plot twist broke me (in the best way) 😭🔥", "true": "Positive", "acceptable": ["positive"]},
    {"id": 47, "text": "it do be like that sometimes", "true": "Neutral", "acceptable": ["neutral"]},
    {"id": 48, "text": "not everyone can be this talented, and clearly it shows here", "true": "Negative", "acceptable": ["negative"]},
    {"id": 49, "text": "I'm obsessed, in a concerning way", "true": "Positive", "acceptable": ["positive"]},
    {"id": 50, "text": "this is fine actually, I've made peace with the chaos", "true": "Neutral/Mixed", "acceptable": ["neutral", "mixed"]}
]

def run_benchmark():
    with TestClient(app) as client:
        texts = [ex["text"] for ex in HARD_50_EXAMPLES]
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"texts": texts, "extract_aspects": True, "enable_fallback": False})
        batch_latency = (time.perf_counter() - t0) * 1000
        
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        
        passed = 0
        failed = []
        
        print("=" * 90)
        print("                HARD SENTIMENT TEST SET (50 EXAMPLES) BENCHMARK                ")
        print("=" * 90)
        
        for i, ex in enumerate(HARD_50_EXAMPLES):
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
                
            print(f"#{ex['id']:02d} | True: {ex['true']:<15} | Pred: {pred_label.upper():<8} ({res['confidence']*100:4.1f}%) | {status} | {ex['text'][:60]}...")
            
        print("=" * 90)
        print(f"RESULTS: {passed} / {len(HARD_50_EXAMPLES)} PASSED ({passed / len(HARD_50_EXAMPLES) * 100:.1f}%)")
        print(f"Total Batch Latency: {batch_latency:.2f} ms ({batch_latency / len(HARD_50_EXAMPLES):.3f} ms / post)")
        print("=" * 90)
        
        if failed:
            print(f"\nFAILED EXAMPLES ({len(failed)}):")
            for f in failed:
                print(f"\n[#{f['id']}] Text: \"{f['text']}\"")
                print(f"       Expected: {f['true']} | Predicted: {f['pred'].upper()} (Conf: {f['confidence']*100:.1f}%)")
                print(f"       Probs: {f['probabilities']} | Reason: {f['reason']}")
                
if __name__ == "__main__":
    run_benchmark()
