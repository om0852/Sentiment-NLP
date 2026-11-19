"""Label posts with an LLM through the local Codex (OpenAI) or Antigravity (Gemini) CLI.

NOTE: this sends post text to OpenAI / Google. Only the cleaned post text is sent (no ids, users, urls fields).
Posts are sent in numbered batches; the model must answer with one JSON object per post. Results are appended
to the output file as each batch finishes, so the script can be stopped and re-run to resume.

Input: a JSONL file with {"id": ..., "text": ...} per line.
Output: JSONL with {"id", "label", "labeller"} (label in negative/neutral/positive, or "skip").

Usage:
    python scripts/llm_label.py --cli codex --input posts.jsonl --output data/silver/llm_codex.jsonl --batch 50
    python scripts/llm_label.py --cli agy   --input posts.jsonl --output data/silver/llm_agy.jsonl   --batch 50
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

GUIDE = """You are labelling the sentiment of social media posts, titles and news items (many are Indian: English,
Hindi, Marathi, Hinglish; some other languages). For each numbered post give exactly one label:

positive = praise, happiness, gratitude, excitement, congratulations, recommendation, good news, enthusiastic promotion
negative = complaint, anger, sadness, criticism, disappointment, mockery, sarcasm, bad news, tragedy/condolence
neutral  = information, questions, announcements, plain listings/specs/prices, hashtag-only posts, schedules
skip     = you genuinely cannot read the language

Rules:
1. News/facts: label the polarity of the event for the general public ("120 killed" negative, "India wins" positive,
   "rate unchanged" neutral).
2. Sarcasm: label the intended meaning ("Great, another 2-hour delay" is negative).
3. Mixed posts: label the overall takeaway; neutral only if both sides truly cancel out.
4. Political posts: label the tone of the post itself (a party celebrating = positive; attacking opponents = negative).
5. Ads: excited promotion = positive; plain listing = neutral.
6. Hashtag/emoji-only: use a clear emoji if present, else neutral.

Answer with ONLY JSON lines, one per post, in order, like:
{"i": 1, "label": "positive"}
{"i": 2, "label": "neutral"}
No other text."""

LABELS = {"positive", "negative", "neutral", "skip"}
AGY = os.path.join(os.environ.get("LOCALAPPDATA", ""), "agy", "bin", "agy.exe")


def build_prompt(batch):
    lines = [GUIDE, "", "Posts:"]
    for i, p in enumerate(batch, 1):
        text = " ".join(p["text"].split())[:500]
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def run_cli(cli, prompt, workdir, timeout):
    prompt_path = os.path.join(workdir, "prompt.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    if cli == "codex":
        out_path = os.path.join(workdir, "out.md")
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(prompt_path, encoding="utf-8") as fin:
            subprocess.run(["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "-o", out_path, "-"],
                           stdin=fin, cwd=workdir, capture_output=True, timeout=timeout, shell=(os.name == "nt"))
        return open(out_path, encoding="utf-8").read() if os.path.exists(out_path) else ""
    res = subprocess.run([AGY, "-p", f"Read the file {prompt_path} and do exactly what it says. Do not modify files.",
                          "--mode", "plan", "--sandbox", "--print-timeout", f"{timeout}s"],
                         cwd=workdir, capture_output=True, timeout=timeout + 60, encoding="utf-8", errors="replace")
    return res.stdout


def parse(answer, n):
    got, dupes = {}, set()
    for m in re.finditer(r'\{[^{}]*"i"\s*:\s*(\d+)[^{}]*"label"\s*:\s*"(\w+)"[^{}]*\}', answer):
        i, label = int(m.group(1)), m.group(2).lower()
        if 1 <= i <= n and label in LABELS:
            if i in got:
                dupes.add(i)  # the model answered this index twice: trust neither
            got[i] = label
    for i in dupes:
        got.pop(i, None)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", choices=["codex", "agy"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        posts = [json.loads(line) for line in f if line.strip()]
    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [p for p in posts if p["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.cli}: {len(posts):,} posts, {len(done):,} done, {len(todo):,} to label", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=f"llmlabel_{args.cli}_")
    t0, labelled, failed_batches = time.time(), 0, 0
    with open(args.output, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), args.batch):
            batch = todo[b: b + args.batch]
            try:
                answer = run_cli(args.cli, build_prompt(batch), workdir, args.timeout)
            except subprocess.TimeoutExpired:
                answer = ""
            got = parse(answer, len(batch))
            if len(got) < len(batch):
                # keep what parsed; the missing posts are retried on the next run
                print(f"  batch {b // args.batch + 1}: {len(got)}/{len(batch)} labels parsed; "
                      f"the rest will be retried on the next run", flush=True)
            if not got:
                failed_batches += 1
                if failed_batches >= 3 and labelled == 0:
                    sys.exit("three failed batches in a row with nothing labelled: check the CLI (auth/quota)")
                continue
            for i, p in enumerate(batch, 1):
                if i in got:
                    out.write(json.dumps({"id": p["id"], "label": got[i], "labeller": args.cli}) + "\n")
            out.flush()
            labelled += len(got)
            rate = labelled / (time.time() - t0)
            print(f"  {labelled:,}/{len(todo):,} labelled ({rate * 60:.0f} posts/min)", flush=True)


if __name__ == "__main__":
    main()
