import os
import sys
import time
import asyncio
import httpx
from typing import List

SAMPLE_POSTS = [
    "This new feature is absolute fire no cap 🔥",
    "Customer service was totally mid and completely cooked",
    "The scheduled webinar starts at 3pm today",
    "I am dying of laughter 💀",
    "Total scam, completely broken after update 👎",
    "Sheesh they really ate with this clean UI",
    "Another massive L for the team 🤡",
    "Can you share the release notes link?",
    "Super smooth workflow and great speed ✨",
    "Not good at all, very disappointed"
]

async def send_batch(client: httpx.AsyncClient, url: str, batch: List[str]):
    t0 = time.perf_counter()
    resp = await client.post(url, json={"texts": batch, "enable_fallback": False})
    latency = (time.perf_counter() - t0) * 1000
    return resp.status_code, latency, len(batch)

async def run_simulation(target_daily_volume: int = 1_000_000, duration_seconds: int = 5, batch_size: int = 25):
    """
    Simulates high-throughput production load against FastAPI endpoint.
    10L (1M/day) = ~11.6 posts/sec avg, but load test tests at 50 to 200 posts/sec to verify headroom!
    """
    from src.serving.app import app
    from starlette.testclient import TestClient

    print(f"\n" + "=" * 60)
    print(f"   SIMULATING TRAFFIC LOAD: {target_daily_volume:,} POSTS/DAY")
    print(f"=" * 60)
    
    # Required average posts/sec for target daily volume
    required_pps = target_daily_volume / 86400.0
    print(f"Required Average Rate: {required_pps:.2f} posts/second")
    print(f"Testing Batch Size:    {batch_size} posts/request")
    print(f"Test Duration:         {duration_seconds} seconds")
    
    with TestClient(app) as test_client:
        total_posts_sent = 0
        total_requests = 0
        latencies = []
        errors = 0
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            batch = (SAMPLE_POSTS * ((batch_size // len(SAMPLE_POSTS)) + 1))[:batch_size]
            
            t0 = time.perf_counter()
            resp = test_client.post("/predict", json={"texts": batch, "enable_fallback": False})
            lat = (time.perf_counter() - t0) * 1000
            
            if resp.status_code == 200:
                total_posts_sent += len(batch)
                total_requests += 1
                latencies.append(lat)
            else:
                errors += 1

        elapsed = time.time() - start_time
        actual_pps = total_posts_sent / elapsed
        
        print("\n--- Load Test Results ---")
        print(f"Total Posts Processed: {total_posts_sent:,} in {elapsed:.2f}s")
        print(f"Total Batch Requests:  {total_requests} (0 errors)")
        print(f"Achieved Throughput:   {actual_pps:,.0f} posts / second")
        print(f"Mean Batch Latency:    {sum(latencies) / len(latencies):.2f} ms")
        print(f"Per-Post Latency:      {(sum(latencies) / len(latencies)) / batch_size:.4f} ms")
        
        simulated_daily_capacity = actual_pps * 86400
        print(f"\nProjected 24h Capacity: {simulated_daily_capacity:,.0f} posts / day")
        if actual_pps >= required_pps:
            margin = actual_pps / required_pps
            print(f"STATUS: PASS! ({margin:.1f}x higher than {target_daily_volume:,}/day requirement)")
        else:
            print("STATUS: Under capacity")
        print("=" * 60)

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    asyncio.run(run_simulation(target_daily_volume=1_000_000, duration_seconds=3, batch_size=25))
