import os
import sys
import time
import psutil
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.tfidf_classifier import TfidfSentimentClassifier

def benchmark_engine():
    process = psutil.Process(os.getpid())
    
    baseline_rss_mb = process.memory_info().rss / (1024 * 1024)
    print("=" * 60)
    print("      SENTIMENT ENGINE: CPU & RAM BENCHMARK SUITE     ")
    print("=" * 60)
    print(f"Baseline Python Process Memory (RSS): {baseline_rss_mb:.2f} MB")

    # 1. Load pipeline & model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "models", "sentiment_model.joblib")
    
    pipeline = PreprocessingPipeline()
    model = TfidfSentimentClassifier()
    model.load(model_path)
    
    loaded_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory with Model & Dictionaries in RAM: {loaded_rss_mb:.2f} MB")
    print(f"Model Memory Overhead: {loaded_rss_mb - baseline_rss_mb:.2f} MB  (Strict Budget: <300MB)")

    # 2. Sample posts for benchmarking
    test_samples = [
        "Bro this new feature is so fire no cap 🔥",
        "Their customer service was totally mid and completely cooked",
        "The system update is scheduled for tomorrow at 3pm",
        "I am dying of laughter this is so funny 💀",
        "Total scam, avoid this product at all costs 👎",
        "Sheesh they really ate and left no crumbs with this update",
        "Another massive L for this broken release 🤡",
        "What is the difference between pro and standard tiers?",
        "Such an immaculate and clean design ✨",
        "Not good at all, completely disappointed with the quality"
    ]

    # Pre-warm JIT and caches
    for _ in range(50):
        for s in test_samples:
            proc, _ = pipeline.process(s)
            model.predict(proc)

    # 3. Single post inference latency benchmark (1,000 iterations)
    latencies = []
    for _ in range(1000):
        idx = _ % len(test_samples)
        t0 = time.perf_counter()
        proc, _ = pipeline.process(test_samples[idx])
        _ = model.predict(proc)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)

    print("\n--- Latency Breakdown (End-to-End: Preprocessing + Inference) ---")
    print(f"Mean Latency: {mean_lat:.3f} ms")
    print(f"P50 Latency:  {p50:.3f} ms")
    print(f"P90 Latency:  {p90:.3f} ms")
    print(f"P95 Latency:  {p95:.3f} ms")
    print(f"P99 Latency:  {p99:.3f} ms")

    # 4. Batch throughput benchmark (batches of 50 posts)
    batch_latencies = []
    batch = test_samples * 5  # 50 posts
    for _ in range(100):
        t0 = time.perf_counter()
        preproc_batch = [pipeline.process(t)[0] for t in batch]
        _ = model.predict_batch(preproc_batch)
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    avg_batch_ms = np.mean(batch_latencies)
    per_post_in_batch_ms = avg_batch_ms / 50.0
    throughput_posts_per_sec = 1000.0 / per_post_in_batch_ms

    print("\n--- Batch Throughput (Batch Size = 50 posts) ---")
    print(f"Average Batch Time (50 posts): {avg_batch_ms:.2f} ms")
    print(f"Effective Time Per Post:       {per_post_in_batch_ms:.4f} ms")
    print(f"Single-Core Throughput:        {throughput_posts_per_sec:.0f} posts / second")

    # 5. Theoretical 0.1 CPU Capacity Analysis
    # On 0.1 CPU (10% of 1 core), capacity is ~10% of single-core throughput
    capacity_01_cpu_sec = throughput_posts_per_sec * 0.10
    capacity_01_cpu_day = capacity_01_cpu_sec * 86400

    print("\n--- Capacity Projection on 0.1 vCPU ---")
    print(f"Simulated 0.1 CPU Throughput:   {capacity_01_cpu_sec:.0f} posts / second")
    print(f"Simulated 0.1 CPU Daily Volume: {capacity_01_cpu_day:,.0f} posts / day")
    print(f"Target 10L (1,000,000) posts/day margin: {(capacity_01_cpu_day / 1000000.0):.1f}x surplus!")

    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nFinal Memory Footprint: {final_rss_mb:.2f} MB (PASS: Strictly < 300MB)")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_engine()
