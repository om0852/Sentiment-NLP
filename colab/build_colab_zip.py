"""Package code + training data for scripts/train_student.py into colab/sentiment_colab.zip.

The zip holds user post text (training data), so it stays in the gitignored colab/ folder.
Usage: SENTIMENT_DATA_ROOT=/checkout/with/data python colab/build_colab_zip.py
"""
import glob
import hashlib
import json
import os
import zipfile

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", repo)
out = os.path.join(repo, "colab", "sentiment_colab.zip")

gold_ids = json.load(open(os.path.join(data_root, "data", "gold", "gold_ids.json")))
# same cache key as scripts/train_student.py for --max-len 128
key = hashlib.md5(f"cardiffnlp/twitter-xlm-roberta-base-sentiment|128|{sorted(set(gold_ids))}".encode()).hexdigest()[:10]
data_files = [
    f"data/silver/token_counts_{key}.json",
    "data/silver/teacher_labels.jsonl",
    "data/silver/llm_codex.jsonl",
    "data/silver/extra_english.jsonl",
    "data/gold/gold_ids.json",
] + [f"data/raw/tweeteval/{s}_{k}.txt" for s in ("train", "val") for k in ("text", "labels")]

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for script in ("train_student.py", "teacher_soft_labels.py"):
        z.write(os.path.join(repo, "scripts", script), f"scripts/{script}")
    for path in glob.glob(os.path.join(repo, "src", "**", "*.py"), recursive=True):
        z.write(path, os.path.relpath(path, repo).replace(os.sep, "/"))
    for rel in data_files:
        src = os.path.join(data_root, rel)
        assert os.path.exists(src), f"missing {src}"
        z.write(src, rel)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
with zipfile.ZipFile(out) as z:
    names = z.namelist()
print(len(names), "files;", [n for n in names if n.startswith("data/")])
