import os
import json
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "")

def inspect_db():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15000)
    
    # 1. Inspect 'cache1' database
    db_cache1 = client["cache1"]
    for col_name in ["listenings", "posts", "keywords"]:
        if col_name in db_cache1.list_collection_names():
            col = db_cache1[col_name]
            count = col.estimated_document_count()
            print(f"\n[cache1.{col_name}] Estimated doc count: {count}")
            sample = list(col.find().limit(1))
            if sample:
                sample[0].pop('_id', None)
                print(f"Sample keys: {list(sample[0].keys())}")
                # print snippet
                preview = {k: (str(v)[:100] if len(str(v)) > 100 else v) for k, v in list(sample[0].items())[:8]}
                print(json.dumps(preview, indent=2, default=str))

    # 2. Inspect 'listening' database
    print("\n--- Inspecting 'listening' database ---")
    db_listening = client["listening"]
    listening_cols = db_listening.list_collection_names()
    print(f"Collections in 'listening' db: {listening_cols}")
    for col_name in listening_cols:
        col = db_listening[col_name]
        count = col.estimated_document_count()
        print(f"\n[listening.{col_name}] Estimated doc count: {count}")
        sample = list(col.find().limit(1))
        if sample:
            sample[0].pop('_id', None)
            print(f"Sample keys: {list(sample[0].keys())}")
            preview = {k: (str(v)[:100] if len(str(v)) > 100 else v) for k, v in list(sample[0].items())[:8]}
            print(json.dumps(preview, indent=2, default=str))

if __name__ == '__main__':
    inspect_db()
