import os
import sys
import json
import time
import argparse
from datetime import datetime
from pymongo import MongoClient

# Ensure UTF-8 console output on Windows
sys.stdout.reconfigure(encoding='utf-8')

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://creatosaurus1:EOaVFfQ5YhOD3UhF@creatosaurus.7trc5.mongodb.net/cache1"
)

def fetch_300k_posts(target_count: int = 300000, output_path: str = None, batch_size: int = 5000):
    if not output_path:
        project_root = r"C:\Users\salun\OneDrive - smarttech\Desktop\D folder\sentiment-engine"
        output_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "posts_300k.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 80)
    print("🚀 PRODUCTION MONGODB DATA EXTRACTOR (300K POSTS)")
    print("=" * 80)
    print(f"Database Target:     creatosaurus.7trc5.mongodb.net/cache1")
    print(f"Collection:          listenings (Total available: ~693,483 docs)")
    print(f"Extraction Target:   {target_count:,} posts")
    print(f"Output File:         {output_path}")
    print(f"Cursor Batch Size:   {batch_size:,} docs/fetch")
    print(f"Start Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_start = time.perf_counter()

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=20000,
            maxPoolSize=20,
            socketTimeoutMS=60000
        )
        db = client["cache1"]
        col = db["listenings"]

        # Optimize projection: only pull necessary fields to minimize network transfer
        projection = {
            "_id": 1,
            "title": 1,
            "content": 1,
            "description": 1,
            "snippet": 1,
            "sentimental": 1,
            "source": 1,
            "createdAt": 1,
            "url": 1
        }

        # Filter out completely empty records
        query = {
            "$or": [
                {"content": {"$exists": True, "$ne": ""}},
                {"title": {"$exists": True, "$ne": ""}},
                {"description": {"$exists": True, "$ne": ""}}
            ]
        }

        cursor = col.find(query, projection).batch_size(batch_size)

        fetched_count = 0
        skipped_empty = 0

        print("\nInitiating streaming cursor and writing JSON array to disk...\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first_entry = True

            for doc in cursor:
                title = str(doc.get("title") or "").strip()
                content = str(doc.get("content") or doc.get("description") or doc.get("snippet") or "").strip()

                if not title and not content:
                    skipped_empty += 1
                    continue

                post_record = {
                    "id": str(doc.get("_id")),
                    "title": title,
                    "description": content,
                    "original_sentiment": str(doc.get("sentimental") or "").strip(),
                    "source": str(doc.get("source") or "").strip(),
                    "created_at": str(doc.get("createdAt") or "").strip(),
                    "url": str(doc.get("url") or "").strip()
                }

                if not first_entry:
                    f.write(",\n")
                else:
                    first_entry = False

                f.write(json.dumps(post_record, ensure_ascii=False))
                fetched_count += 1

                # Progress logging every 10,000 posts
                if fetched_count % 10000 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = fetched_count / elapsed if elapsed > 0 else 0
                    remaining = (target_count - fetched_count) / rate if rate > 0 else 0
                    file_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"[{fetched_count:>7,}/{target_count:,}] ({(fetched_count/target_count)*100:5.1f}%) | Speed: {rate:>6.0f} posts/sec | Elapsed: {elapsed:5.1f}s | ETA: {remaining/60:4.1f} min | File Size: {file_mb:6.1f} MB")

                if fetched_count >= target_count:
                    break

            f.write("\n]\n")

        total_elapsed = time.perf_counter() - t_start
        final_mb = os.path.getsize(output_path) / (1024 * 1024)
        avg_rate = fetched_count / total_elapsed if total_elapsed > 0 else 0

        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 80)
        print(f"Total Posts Saved:   {fetched_count:,}")
        print(f"Skipped Empty Docs:  {skipped_empty:,}")
        print(f"Total Execution Time:{total_elapsed:5.1f} seconds ({total_elapsed/60:4.2f} minutes)")
        print(f"Average Throughput:  {avg_rate:,.0f} posts / second")
        print(f"Final File Path:     {output_path}")
        print(f"Final JSON Size:     {final_mb:.2f} MB")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Fetch 300,000 posts from MongoDB to JSON")
    parser.add_argument("--limit", type=int, default=300000, help="Number of posts to fetch (default: 300,000)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=5000, help="MongoDB cursor batch size (default: 5,000)")
    args = parser.parse_args()

    fetch_300k_posts(target_count=args.limit, output_path=args.output, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
