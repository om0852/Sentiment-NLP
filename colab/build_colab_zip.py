"""Package code + training data for scripts/train_student.py into a zip for Google Colab.

The zip holds user post text (training data), so it stays in the gitignored colab/ folder.
Usage:
    SENTIMENT_DATA_ROOT=/checkout/with/data python colab/build_colab_zip.py                 # v2/v3 package
    SENTIMENT_DATA_ROOT=/checkout/with/data LABELLED=data/silver/claude_listening_labels.jsonl \
        OUT=sentiment_colab_v4.zip python colab/build_colab_zip.py                           # + directly labelled posts
"""
import glob
import hashlib
import json
import os
import zipfile

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.environ.get("SENTIMENT_DATA_ROOT", repo)
out = os.path.join(repo, "colab", os.environ.get("OUT", "sentiment_colab.zip"))
labelled = [p.replace("\\", "/") for p in os.environ.get("LABELLED", "").split(",") if p]
for p in labelled:
    # stored in the zip at this same relative path, which the notebook passes to --labelled
    assert p.startswith("data/") and ".." not in p.split("/"), f"LABELLED paths must be relative, under data/: {p}"


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


gold_ids = json.load(open(os.path.join(data_root, "data", "gold", "gold_ids.json")))
# same vocabulary-cache key as scripts/train_student.py for --max-len 128 (+ the labelled corpus, if any)
corpus_id = "|".join(f"{os.path.basename(p)}:{file_md5(os.path.join(data_root, p))}" for p in labelled)
key_src = f"cardiffnlp/twitter-xlm-roberta-base-sentiment|128|{sorted(set(gold_ids))}" + (f"|{corpus_id}" if corpus_id else "")
key = hashlib.md5(key_src.encode()).hexdigest()[:10]
data_files = [
    f"data/silver/token_counts_{key}.json",
    "data/silver/teacher_labels.jsonl",
    "data/silver/llm_codex.jsonl",
    "data/gold/gold_ids.json",
] + labelled + [f"data/raw/tweeteval/{s}_{k}.txt" for s in ("train", "val") for k in ("text", "labels")]
if not labelled:
    data_files.append("data/silver/extra_english.jsonl")

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for script in ("train_student.py", "teacher_soft_labels.py"):
        z.write(os.path.join(repo, "scripts", script), f"scripts/{script}")
    for path in glob.glob(os.path.join(repo, "src", "**", "*.py"), recursive=True):
        z.write(path, os.path.relpath(path, repo).replace(os.sep, "/"))
    for rel in data_files:
        src = os.path.join(data_root, rel)
        assert os.path.exists(src), f"missing {src} (for the vocabulary cache, run train_student.py once locally)"
        z.write(src, rel)
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
with zipfile.ZipFile(out) as z:
    names = z.namelist()
print(len(names), "files;", [n for n in names if n.startswith("data/")])
