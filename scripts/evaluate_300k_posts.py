import os
import sys
import json
import time
import argparse
from collections import Counter
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
sys.path.insert(0, project_root)

# Global worker state
_pipeline = None
_extractor = None
_analyzer = None
_clf = None

def init_worker(model_path: str):
    global _pipeline, _extractor, _analyzer, _clf
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.preprocessing.aspect_extractor import AspectExtractor
    from src.preprocessing.context_analyzer import ContextAnalyzer
    from src.models.tfidf_classifier import TFIDFClassifier

    _pipeline = PreprocessingPipeline()
    _extractor = AspectExtractor()
    _analyzer = ContextAnalyzer()
    _clf = TFIDFClassifier.load_model(model_path)

def process_chunk_worker(chunk_tuple):
    chunk_id, chunk = chunk_tuple
    global _pipeline, _extractor, _analyzer, _clf

    raw_texts = []
    processed_texts = []
    for p in chunk:
        t = p.get("title") or ""
        d = p.get("description") or ""
        ft = f"{t}. {d}".strip() if t and d else (d or t)
        raw_texts.append(ft)
        pt, _ = _pipeline.process(ft)
        processed_texts.append(pt)

    preds = _clf.predict_batch(processed_texts, confidence_threshold=0.60)

    results = []
    for idx, p in enumerate(chunk):
        raw_t = raw_texts[idx]
        pred = preds[idx]
        aspects = _extractor.extract_aspects(raw_t)
        final_label, conf, _, reason = _analyzer.analyze(
            raw_t, str(pred["label"]).lower(), float(pred["confidence"]), aspects=aspects
        )
        db_sent = str(p.get("original_sentiment") or "").strip().lower()
        results.append({
            "final_label": final_label,
            "confidence": round(conf, 3),
            "reason": reason,
            "aspects": aspects,
            "db_sentiment": db_sent,
            "id": p.get("id"),
            "text": raw_t[:150],
            "source": p.get("source")
        })

    return chunk_id, results

def evaluate_300k_posts(input_path: str = None, sample_limit: int = None, chunk_size: int = 4000, workers: int = 6):
    if not input_path:
        input_path = os.path.join(project_root, "data", "raw", "posts_300k.json")

    model_path = os.path.join(project_root, "models", "sentiment_model.joblib")
    report_path = os.path.join(project_root, "data", "evaluation_300k_report.json")

    print("=" * 80)
    print("🚀 PRODUCTION SENTIMENT NLP EVALUATION ON 300K REAL MONGODB POSTS")
    print("=" * 80)
    print(f"Dataset Input:       {input_path}")
    print(f"Chunk Size:          {chunk_size:,} posts / parallel chunk")
    print(f"Parallel Workers:    {workers} CPU processes")
    if sample_limit:
        print(f"Sample Limit:        {sample_limit:,} posts")
    print(f"Start Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("Reading dataset into memory...", flush=True)
    t0_read = time.perf_counter()
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    if sample_limit and sample_limit < len(posts):
        posts = posts[:sample_limit]

    total_posts = len(posts)
    print(f"Loaded {total_posts:,} posts in {time.perf_counter() - t0_read:.2f}s.\n")

    chunks = [(i, posts[i:i + chunk_size]) for i in range(0, total_posts, chunk_size)]
    total_chunks = len(chunks)
    print(f"Queued {total_chunks} chunks across {workers} workers. Starting parallel inference...", flush=True)

    sentiment_counts = Counter()
    legacy_db_counts = Counter()
    aspect_counts = Counter()
    aspect_polarity_counts = {
        "ui_ux": Counter(),
        "performance": Counter(),
        "customer_support": Counter(),
        "pricing_value": Counter(),
        "features": Counter()
    }
    reasons_counts = Counter()

    agreement_with_db = 0
    refined_from_db = 0
    total_evaluated = 0

    sample_refined_cases = []
    sample_mixed_cases = []
    sample_complaint_cases = []

    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = {executor.submit(process_chunk_worker, c): c[0] for c in chunks}

        for future in as_completed(futures):
            _, results = future.result()

            for item in results:
                final_label = item["final_label"]
                conf = item["confidence"]
                reason = item["reason"]
                aspects = item["aspects"]
                db_sent = item["db_sentiment"]

                total_evaluated += 1
                sentiment_counts[final_label] += 1
                legacy_db_counts[db_sent or "unlabeled"] += 1
                reasons_counts[reason] += 1

                for asp, pol in aspects.items():
                    aspect_counts[asp] += 1
                    if asp in aspect_polarity_counts:
                        aspect_polarity_counts[asp][pol] += 1

                # Legacy DB comparison
                if db_sent and db_sent in ["positive", "negative", "neutral"]:
                    if final_label == db_sent:
                        agreement_with_db += 1
                    else:
                        refined_from_db += 1
                        if len(sample_refined_cases) < 15 and len(item["text"]) > 25:
                            sample_refined_cases.append({
                                "id": item["id"],
                                "source": item["source"],
                                "text": item["text"],
                                "db_sentiment": db_sent.upper(),
                                "model_sentiment": final_label.upper(),
                                "confidence": conf,
                                "reason": reason,
                                "aspects": aspects
                            })

                if final_label == "mixed" and len(sample_mixed_cases) < 10:
                    sample_mixed_cases.append({
                        "id": item["id"],
                        "text": item["text"],
                        "aspects": aspects,
                        "reason": reason
                    })

                if final_label == "negative" and db_sent != "negative" and len(sample_complaint_cases) < 10:
                    sample_complaint_cases.append({
                        "id": item["id"],
                        "text": item["text"],
                        "db_sentiment": db_sent.upper(),
                        "reason": reason,
                        "aspects": aspects
                    })

            elapsed = time.perf_counter() - t_start
            rate = total_evaluated / elapsed if elapsed > 0 else 0
            pct = (total_evaluated / total_posts) * 100
            eta = (total_posts - total_evaluated) / rate if rate > 0 else 0
            print(f"[{total_evaluated:>7,}/{total_posts:,}] ({pct:5.1f}%) | Throughput: {rate:>7,.0f} posts/sec | Elapsed: {elapsed:5.1f}s | ETA: {eta:4.1f}s", flush=True)

    total_elapsed = time.perf_counter() - t_start
    overall_throughput = total_evaluated / total_elapsed if total_elapsed > 0 else 0

    # Print Scorecard
    print("\n" + "=" * 80)
    print("📊 300,000 POSTS EVALUATION SCORECARD & REAL-WORLD RESULTS")
    print("=" * 80)
    print(f"Total Posts Evaluated: {total_evaluated:,}")
    print(f"Total Time Taken:      {total_elapsed:5.2f} seconds ({total_elapsed/60:4.2f} minutes)")
    print(f"Overall Throughput:    {overall_throughput:,.0f} posts / second")
    print(f"Average Latency:       {(total_elapsed/total_evaluated)*1000:5.3f} ms / post")

    print("\n📈 MODEL SENTIMENT DISTRIBUTION (300K REAL POSTS):")
    for label, count in sentiment_counts.most_common():
        pct = (count / total_evaluated) * 100
        print(f"  • {label.upper():<10} {count:>7,} posts ({pct:5.2f}%)")

    print("\n🔄 COMPARISON AGAINST LEGACY MONGODB LABELS:")
    db_labeled_total = agreement_with_db + refined_from_db
    if db_labeled_total > 0:
        print(f"  • Confirmed Exact Alignment:  {agreement_with_db:>7,} posts ({(agreement_with_db/db_labeled_total)*100:5.2f}%)")
        print(f"  • Refined / Corrected Labels: {refined_from_db:>7,} posts ({(refined_from_db/db_labeled_total)*100:5.2f}%)")

    print("\n🏷️ ASPECT MENTIONS ACROSS SOCIAL LISTENING STREAM:")
    for asp, count in aspect_counts.most_common():
        pos_c = aspect_polarity_counts[asp]["positive"]
        neg_c = aspect_polarity_counts[asp]["negative"]
        neu_c = aspect_polarity_counts[asp]["neutral"]
        print(f"  • {asp:<18} {count:>6,} mentions (Pos: {pos_c:,} | Neg: {neg_c:,} | Neu: {neu_c:,})")

    print("\n⚙️ DECISION ENGINE REASONING BREAKDOWN:")
    for r, count in reasons_counts.most_common(10):
        print(f"  • {r:<30} {count:>7,} posts ({(count/total_evaluated)*100:5.2f}%)")

    # Save JSON Report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluated": total_evaluated,
        "elapsed_seconds": round(total_elapsed, 2),
        "throughput_posts_per_sec": round(overall_throughput, 1),
        "avg_latency_ms_per_post": round((total_elapsed / total_evaluated) * 1000, 4),
        "sentiment_distribution": dict(sentiment_counts),
        "legacy_db_distribution": dict(legacy_db_counts),
        "alignment_with_db": {
            "exact_alignment_count": agreement_with_db,
            "refined_count": refined_from_db,
            "alignment_pct": round((agreement_with_db / db_labeled_total) * 100, 2) if db_labeled_total else 0
        },
        "aspect_breakdown": {
            asp: dict(aspect_polarity_counts[asp]) for asp in aspect_polarity_counts
        },
        "decision_reasons": dict(reasons_counts),
        "samples_refined_from_db": sample_refined_cases,
        "samples_mixed": sample_mixed_cases,
        "samples_complaints_caught": sample_complaint_cases
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Full 300k evaluation report saved to: '{report_path}'")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 300,000 MongoDB posts")
    parser.add_argument("--input", type=str, default=None, help="Input posts_300k.json path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-run")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Chunk size per task (default: 4000)")
    parser.add_argument("--workers", type=int, default=6, help="Worker count (default: 6)")
    args = parser.parse_args()

    evaluate_300k_posts(input_path=args.input, sample_limit=args.limit, chunk_size=args.chunk_size, workers=args.workers)

if __name__ == "__main__":
    main()
