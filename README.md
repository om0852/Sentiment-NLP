# ⚡ Ultra-Lightweight Pragmatic Sentiment Engine

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![RAM Footprint](https://img.shields.io/badge/RAM-281MB%20peak-success.svg)]()
[![CPU](https://img.shields.io/badge/CPU-1%20core%20(~870k%20posts%2Fday)-success.svg)]()
[![Cost](https://img.shields.io/badge/Cloud%20Cost-%240.00%2Fmo%20(Free%20Tier)-brightgreen.svg)]()
[![TweetEval](https://img.shields.io/badge/TweetEval%20(held--out)-67.1%25-orange.svg)]()
[![300k Evaluation](https://img.shields.io/badge/Production%20300k%20Stream-923%20posts%2Fsec-blue.svg)]()
[![Live Deployment](https://img.shields.io/badge/Railway-Live%20Production-success.svg)](https://sentiment-nlp-production.up.railway.app/)

A production-grade, ultra-lightweight sentiment analysis microservice specialized in modern social vernacular (**Gen-Z slang, multi-modal emojis, complex negations, contrastive clauses, Hinglish, cultural disaster tropes, and thread-context sarcasm**).

Runs in **~280 MB RAM** on **one CPU core** (~870k posts/day with the transformer tier; 10M+/day with `DISABLE_TRANSFORMER=1`), e.g. on an Oracle Cloud Always Free ARM instance.

---

## 🌟 Live Demo & Interactive Dashboard

- **Live Microservice URL:** [`https://sentiment-nlp-production.up.railway.app/`](https://sentiment-nlp-production.up.railway.app/)
- **Interactive Glassmorphic Playground:** Served directly at root `GET /`
- **Health Check Endpoint:** `GET /health`
- **Real-Time Telemetry & Cache Stats:** `GET /metrics`

---

## 📊 Accuracy & Performance (measured 2026-11-18)

Two tiers: the original TF-IDF model + hand-written rules, and a distilled multilingual transformer
(12-layer MiniLM, 128 tokens, int8 ONNX) trained on ~16k of our real posts labelled by an LLM (Codex, ~89% agreement
with careful manual labels) plus TweetEval. Almost every post goes to the transformer. The fast tier keeps posts under
25 characters, romanized Hinglish/Maranglish, posts where a hand-written rule fires, and posts it is ≥90% sure about
(e.g. dictionary-backed Hindi/Marathi). See [docs/ACCURACY_PIPELINE.md](docs/ACCURACY_PIPELINE.md).

| Accuracy (3-class, data no model was trained on) | Original | v1 (PR #1) | **Now** |
| :--- | :--- | :--- | :--- |
| **Real posts from our data** (182 random posts, labelled per the guide by Claude, not yet by humans) | 41.2% | 59.9% | **78.6%** |
| ↳ on posts not flagged `uncertain` (80% of posts) | n/a | 64.1% | **84.9%** |
| **TweetEval sentiment test** (12,284 tweets, human labels) | 59.4% | 63.4% | **67.1%** |
| ↳ on posts not flagged `uncertain` (76% of posts) | n/a | 69.9% | **73.0%** |
| Hand-written real-world style sets (174 + 126 posts) | 66.7% / 62.7% | 67.2% / 65.1% | 68.4% / 63.5% |
| Hard sets (`benchmark_hard_50.py`, `benchmark_hard_set_2.py`) | 38% / 44% | 42% / 32% | 40% / 20% |

| Resource | Measured (laptop CPU, 1 thread) |
| :--- | :--- |
| Runtime memory (RSS, both tiers loaded) | **275 MB at startup, 281 MB peak** |
| CPU per post (real-post mix) | **~100 ms** (TF-IDF tier alone: ~1 ms) |
| Capacity | **~870k posts/day per CPU core**. Fits one Oracle Always Free Ampere core (1 OCPU / 6 GB, see [deploy/oracle](deploy/oracle/README.md)); `DISABLE_TRANSFORMER=1` gives 10M+/day at lower accuracy |
| Model files on disk | 45 MB transformer + 4 MB tokenizer + 1 MB news detector + 1 MB TF-IDF |

> **Note on accuracy numbers (corrected 2026-11-17).** Earlier versions of this README claimed 100/100 on the hard
> benchmarks and 73.16% on TweetEval. The hard-benchmark result does not reproduce on the committed code, and the
> TweetEval figure came from a model that had been trained on the TweetEval test split. Training now excludes that
> split, and every number above is measured on data no model was trained on.
>
> The number to trust most will be the **human-labelled gold set** (`data/gold/gold_sample.csv`, 1,500 random real
> posts, see [docs/LABELLING_GUIDE.md](docs/LABELLING_GUIDE.md)); run `python scripts/evaluate_gold.py` once it's labelled.
>
> **Known weak spots:** sarcasm/meme templates (hard set 2: 20%) and some short slang the transformer reads as
> neutral, e.g. "Uninstalling immediately #BoycottApp" (the one failing test in `tests/`).

### New response fields

| Field | Meaning |
| :--- | :--- |
| `tier` | `tfidf` (fast tier) or `transformer` |
| `content_type` | `news` for factual reporting / headlines, else `social` |
| `uncertain` | Transformer's calibrated confidence is below `uncertain_below` (0.6). Route to a person or an LLM |

Confidence is now **calibrated** (temperature scaling), so it is lower than before for the TF-IDF tier: 0.6 really means
about 60% likely to be right. A side effect: more posts fall under the default `confidence_threshold` of 0.60 and get
`fallback_required: true` (about 20–30% of posts, vs ~21% before). If you call the API
with `enable_fallback: true`, that means more calls to the fallback service.

---

## 🚀 Core Architectural Features

1. **Option 1: Cultural Disaster & Triumph Metaphors (`culture_tropes_dict.json`)**
   - Resolves pop-culture and tech lore that lack literal sentiment words (e.g. *"They pulled a Season 8 Game of Thrones on this update"*, *"feeling like a 19th-century chimney sweep"*, *"CrowdStrike update vibes"* $\to$ **Negative**; *"UI redesign is literally the Mona Lisa"* $\to$ **Positive**).
   - Includes litotes/negation guards (e.g. *"not a dumpster fire"* avoids being wrongly tagged as a disaster).

2. **Option 2: Contextual Thread / Parent Post Awareness (`PostInput(text, context)`)**
   - Inverts praise into sarcasm when posted under disaster contexts (e.g. Reply: *"Brilliant job guys 👏"*, Context: *"Major database outage"* $\to$ **Negative (`context_mismatch_sarcasm`)**).
   - Thread-aware LRU cache keys prevent collisions between identical text with and without context.

3. **Multi-Modal Contextual Emoji Tokenizer (`emoji_mapper.py`)**
   - Dynamically decodes ambiguous emojis based on syntactic context (e.g. `💀` with delay words $\to$ sarcasm/agony; `💀` with laughter words $\to$ high humor; `🙃` / `🫠` with complaints $\to$ suppressed frustration).

4. **Clause-Level Aspect-Based Sentiment Analysis (ABSA) (`aspect_extractor.py`)**
   - Automatically segments multi-clause feedback across 5 dimensions: **UI/UX**, **Performance**, **Support**, **Pricing**, and **Features**.

5. **Sub-0.07ms High-Speed LRU Cache (`cache.py`)**
   - In-memory 10,000-entry LRU cache delivering $0.069\text{ ms}$ response times for viral posts and repetitive social traffic.

---

## 🛠️ API Reference

### 1. Simple Post Analysis (`texts` array)
```bash
curl -X POST "https://sentiment-nlp-production.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Not gonna lie, I thought this phone would be mid but ngl it is actually not bad at all, kinda impressed ngl",
      "They really pulled a Season 8 Game of Thrones on this update"
    ]
  }'
```

#### Response:
```json
{
  "results": [
    {
      "text": "Not gonna lie, I thought this phone would be mid but ngl it is actually not bad at all, kinda impressed ngl",
      "label": "positive",
      "confidence": 0.88,
      "probabilities": { "positive": 0.88, "negative": 0.09, "neutral": 0.03 },
      "aspects": { "Features": "positive" },
      "fallback_required": false,
      "reason": "praise_shift_detected",
      "cached": false
    },
    {
      "text": "They really pulled a Season 8 Game of Thrones on this update",
      "label": "negative",
      "confidence": 0.92,
      "probabilities": { "negative": 0.92, "positive": 0.06, "neutral": 0.02 },
      "aspects": {},
      "fallback_required": false,
      "reason": "cultural_disaster_metaphor",
      "cached": false
    }
  ],
  "total_processed": 2,
  "fallback_count": 0,
  "cache_hits": 0,
  "batch_latency_ms": 1.74,
  "model_version": "tfidf-context-absa-v4"
}
```

---

### 2. Contextual Thread Post Analysis (`posts` array with parent context)
```bash
curl -X POST "https://sentiment-nlp-production.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "posts": [
      {
        "text": "Brilliant job guys, love to see it 👏",
        "context": "Major database outage affecting all production servers"
      }
    ]
  }'
```

#### Response:
```json
{
  "results": [
    {
      "text": "Brilliant job guys, love to see it 👏",
      "context": "Major database outage affecting all production servers",
      "label": "negative",
      "confidence": 0.92,
      "probabilities": { "negative": 0.92, "positive": 0.06, "neutral": 0.02 },
      "aspects": {},
      "fallback_required": false,
      "reason": "context_mismatch_sarcasm",
      "cached": false
    }
  ],
  "total_processed": 1,
  "fallback_count": 0,
  "cache_hits": 0,
  "batch_latency_ms": 1.48,
  "model_version": "tfidf-context-absa-v4"
}
```

---

## 🧪 Benchmark Verification

To run the complete benchmark suite locally:

```bash
# 1. Activate environment
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# 2. Run Hard Set 1 (Slang, Negation, Litotes - 50 samples)
python scripts/benchmark_hard_50.py

# 3. Run Hard Set 2 (Memes, Sarcasm, Pop-Culture Lore - 50 samples)
python scripts/benchmark_hard_set_2.py

# 4. Run the held-out TweetEval benchmark (data/ is gitignored; point at the checkout that has it)
SENTIMENT_DATA_ROOT=/path/to/checkout/with/data python scripts/benchmark_tweeteval.py

# 5. Run automated test suite
pytest tests/
```

---

## 🐳 Docker Deployment

```bash
# Build the lightweight container
docker build -t sentiment-engine:latest -f docker/Dockerfile .

# Run with strictly enforced resource quotas
docker run -d \
  -p 8000:8000 \
  --memory=300m \
  --cpus=0.1 \
  --name sentiment-service \
  sentiment-engine:latest
```

---

## 📚 Detailed Documentation

- **[System Architecture Deep Dive](docs/ARCHITECTURE.md):** Complete architectural overview, dual-stage pipeline diagrams, memory profiling, and mathematical capacity proofs under $0.1\text{ vCPU}$.
- **[Implementation & Operational Guide](docs/IMPLEMENTATION.md):** Module breakdown, schema definitions, edge case solutions, and operational runbooks.

