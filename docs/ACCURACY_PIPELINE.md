# Accuracy pipeline: how the transformer tier is built and measured

Everything here runs locally. Nothing writes to MongoDB or any other database; the only input is the local
export `data/raw/posts_300k.json`. `data/` is gitignored because it holds user posts. Point
`SENTIMENT_DATA_ROOT` at the checkout that has it.

```
posts_300k.json ──► build_silver_dataset.py ──► data/gold/gold_sample.csv   (1,500 posts for humans, never trained on)
                                              └► data/silver/teacher_labels.jsonl (60k posts labelled by a local teacher)
teacher_labels.jsonl + TweetEval train/val ──► train_student.py ──► models/student/ (int8 ONNX + trimmed tokenizer)
posts_300k.json ──► train_news_detector.py ──► models/news_detector.npz
everything ──► evaluate_cascade.py --tune --write-config ──► models/cascade_config.json + benchmark table
gold_sample.csv (once humans label it) ──► evaluate_gold.py ──► the accuracy number to publish
```

## Steps

```bash
export SENTIMENT_DATA_ROOT=/path/to/checkout/with/data

# 1 + 2. Gold sample for humans, and teacher labels for 60k other posts (local XLM-R, ~2-3 h on a laptop CPU)
python scripts/build_silver_dataset.py --pool 60000

# 3. Distil a 6-layer multilingual MiniLM student (~1.5-2 h on a 4-core laptop CPU), export int8 ONNX
python scripts/train_student.py --epochs 1.5 --layers 6 --max-len 64
#    resume after an interruption from a saved state_dict:
#    python scripts/train_student.py --epochs 0.9 --lr 4e-5 --layers 6 --max-len 64 --resume path/to/best.pt

# 6. News / factual-reporting detector (RSS vs social posts)
python scripts/train_news_detector.py

# 4, 5, 7. Calibrate, ablate rules, choose routing; then benchmark every set
python scripts/evaluate_cascade.py --tune --write-config --max-escalation 1.0   # transformer for every post
#    or let the CPU budget decide how many posts reach the transformer:
#    python scripts/evaluate_cascade.py --tune --write-config --posts-per-day 1000000 --vcpu 0.1

# Human labels: gold set (docs/LABELLING_GUIDE.md), plus the active-learning queue of hard posts
python scripts/evaluate_gold.py
python scripts/export_review_queue.py --n 600
```

## What each piece is for

| Piece | File | Why |
|---|---|---|
| Post cleaning | `src/preprocessing/post_text.py` | Strips search highlighting, HTML entities, platform suffixes and "Photo by…" boilerplate so models learn tone, not source |
| Teacher labels | `scripts/build_silver_dataset.py` | In-domain training data without sending posts anywhere. Only posts where the teacher and the stored DB label agree are trained on |
| Student | `scripts/train_student.py`, `src/models/student_onnx.py` | Multilingual sentence-level model. Vocabulary and tokenizer trimmed 250k → 60k pieces (covers 99.6% of tokens), 12 → 6 layers, int8: ~34 MB on disk, ~130 MB RAM |
| Calibration | `src/models/calibration.py` | Temperature scaling so confidence means what it says. Fitted on held-out data for both tiers |
| Cascade | `src/serving/cascade.py` | One function, `classify_batch`, used by both the API and the evaluation scripts, so we measure exactly what we serve |
| News detector | `src/models/news_detector.py` | Hashed linear model, pure NumPy (scikit-learn alone would cost ~80 MB RAM). Sets `content_type` |
| Rule pruning | `evaluate_cascade.py` → `disabled_rules` | Each rule is scored on ~33k real posts: how often it fixes vs breaks the model. Rules with clear net harm are switched off in config, not deleted |
| Review queue | `scripts/export_review_queue.py` | Posts where teacher and DB disagree or the teacher is unsure. The best use of human labelling time |

## Serving switches

| Env var / config | Effect |
|---|---|
| `DISABLE_TRANSFORMER=1` | TF-IDF + rules only (old behaviour, ~3 ms/post) |
| `TRANSFORMER_THREADS` | onnxruntime threads (default 1) |
| `models/cascade_config.json` → `escalate_below` | `1.01` = transformer for every post; lower values send only low-confidence posts |
| `uncertain_below` | Transformer confidence below this sets `uncertain: true` |

## Known limits (read before quoting numbers)

- Silver labels come from the teacher model, so "silver validation" accuracy mostly measures agreement with
  the teacher. Quote the gold set (human labels) or an independent set instead.
- The teacher itself is roughly 65% accurate on our real posts. A student distilled from it cannot do much
  better. Beating that needs human labels: the gold set and the review queue.
- TweetEval test is held out from all training, so it's a fair (if dated, English-only) benchmark.
