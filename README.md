# ⚡ Ultra-Lightweight Pragmatic Sentiment Engine

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![RAM Footprint](https://img.shields.io/badge/RAM-549MB%20peak-success.svg)]()
[![CPU](https://img.shields.io/badge/CPU-1%20core%20(~870k%20posts%2Fday)-success.svg)]()
[![Cost](https://img.shields.io/badge/Cloud%20Cost-%240.00%2Fmo%20(Free%20Tier)-brightgreen.svg)]()
[![TweetEval](https://img.shields.io/badge/TweetEval%20(held--out)-71.6%25-orange.svg)]()
[![300k Evaluation](https://img.shields.io/badge/Production%20300k%20Stream-923%20posts%2Fsec-blue.svg)]()
[![Live Deployment](https://img.shields.io/badge/Railway-Live%20Production-success.svg)](https://sentiment-nlp-production.up.railway.app/)

A production-grade, ultra-lightweight sentiment analysis microservice specialized in modern social vernacular (**Gen-Z slang, multi-modal emojis, complex negations, contrastive clauses, Hinglish, cultural disaster tropes, and thread-context sarcasm**).

Runs in **~550 MB RAM** on **one CPU core** (~870k posts/day with all tiers; 10M+/day with `DISABLE_TRANSFORMER=1`), e.g. on an Oracle Cloud Always Free ARM instance.

---

## 🌟 Live Demo & Interactive Dashboard

- **Live Microservice URL:** [`https://sentiment-nlp-production.up.railway.app/`](https://sentiment-nlp-production.up.railway.app/)
- **Interactive Glassmorphic Playground:** Served directly at root `GET /`
- **Health Check Endpoint:** `GET /health`
- **Real-Time Telemetry & Cache Stats:** `GET /metrics`

---

## 📊 Accuracy & Performance (measured 2026-11-18)

Three tiers, chosen per post by `src/serving/cascade.py`:

| Tier | Model | Gets |
| :--- | :--- | :--- |
| `transformer` | 12-layer multilingual MiniLM (int8 ONNX, 45 MB), v4: trained on ~35k listening posts across all clients and platforms labelled by Claude, ~16k real posts labelled by Codex, and TweetEval | Most posts: long posts, news, Hindi/Marathi and other scripts |
| `english_expert` | `cardiffnlp/twitter-roberta-base-sentiment-latest` (int8 ONNX, 125 MB in 3 parts) | Short English social posts (< 140 chars, Latin script, not Hinglish, not news) |
| `tfidf` | Original TF-IDF + hand-written rules | Posts where a rule fires, romanized Hinglish, tiny posts, posts it is ≥90% sure about |

See [docs/ACCURACY_PIPELINE.md](docs/ACCURACY_PIPELINE.md) for how each piece is built and measured.

| Accuracy (3-class, held-out data) | Original | v1 | v2 | **Now** | roberta-latest alone | tabularisai alone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real posts from our data** (182 random, labelled per the guide by Claude) | 41.2% | 59.9% | 78.6% | **78.6%** | 59.9% | 54.4% |
| ↳ Hindi/Marathi script (33) | — | — | 78.8% | **78.8%** | 39.4% | 57.6% |
| **Real posts, Codex labels** (651 never trained on) | 43.5% | 58.5% | 74.3% | **76.0%** | 63.6% | 49.6% |
| ↳ Hindi/Marathi script (119) | — | — | 81.5% | **81.5%** | 34.5% | 44.5% |
| **TweetEval sentiment test** (12,284, human labels) | 59.4% | 63.4% | 67.3% | **71.6%** | 72.2% | 57.5% |
| Hand-written real-world style sets (174 / 126) | 66.7% / 62.7% | 67.2% / 65.1% | 69.0% / 61.9% | **78.2% / 73.0%** | 76.4% / 70.6% | 54.0% / 61.1% |
| Hard sets (`benchmark_hard_50.py` / `_set_2.py`) | 38% / 44% | 42% / 32% | 40% / 20% | **54%** / 36% | 50% / 46% | 42% / 42% |

Reproduce with `python scripts/benchmark_external_models.py` (it downloads the two comparison models).
The table above was measured with the v2 transformer; v4 results follow.

**v4 transformer (2026-11-19).** Retrained on ~38k fresh listening posts (Apr–May 2026, every client and platform),
labelled by Claude agents per the labelling guide, not by the stored library labels (which agree with Claude only 51% of the time).
Two independent Claude labellers agreed on 93.5% of a 2,000-post pilot. Scored on 1,000 random posts from the same
export that were held out of training:

| Accuracy, 3-class | v2 | **v4** |
| :--- | :--- | :--- |
| **Fresh listening posts, all clients** (1,000 held out), light mode as deployed | 70.8% | **80.4%** |
| ↳ same, full mode (+ English expert) | 71.1% | 78.6% |
| Older listening export (142, Claude labels), light | 74.6% | 76.1% |
| Real posts, Codex labels (651), light / full | 74.5% / 76.2% | 75.6% / 77.3% |
| Real posts, Claude labels (182), light / full | 78.0% / 77.5% | 76.9% / 76.9% |
| TweetEval test (12,284), light / full | 67.2% / 71.4% | 67.2% / 71.4% |
| Hard sets 1 / 2, full | 52% / 38% | 52% / 40% |

Most of v4's gain comes from fixing posts that v2 wrongly called neutral: positive→neutral errors fell from 128 to 63 and negative→neutral from 73 to 41.
In this table, a prediction counts as right when it matches any label the set allows. That rule differs slightly from the table above,
so compare numbers within a table, not across tables.

| Resource (laptop CPU, 1 thread, 600 random real posts) | Measured |
| :--- | :--- |
| Runtime memory (RSS, all tiers loaded) | **536 MB at startup, 549 MB peak** |
| CPU per post | **~100 ms** (83% transformer, 14% English expert, 3% TF-IDF) |
| Capacity | **~870k posts/day per CPU core**. Fits an Oracle Always Free Ampere core (1 OCPU / 6 GB, see [deploy/oracle](deploy/oracle/README.md)) |
| Lighter modes | `DISABLE_ENGLISH_EXPERT=1` (~280 MB); `DISABLE_TRANSFORMER=1` (TF-IDF only, ~1 ms/post, 10M+/day, much lower accuracy) |

> **Note on accuracy numbers (corrected 2026-11-17).** Earlier versions of this README claimed 100/100 on the hard
> benchmarks and 73.16% on TweetEval. The hard-benchmark result does not reproduce on the committed code, and the
> TweetEval figure came from a model that had been trained on the TweetEval test split. Training now excludes that
> split, and every number above is measured on data no model was trained on.
>
> The number to trust most will be the **human-labelled gold set** (`data/gold/gold_sample.csv`, 1,500 random real
> posts, see [docs/LABELLING_GUIDE.md](docs/LABELLING_GUIDE.md)); run `python scripts/evaluate_gold.py` once it's labelled.
>
> **Known weak spot:** sarcasm/meme templates (hard set 2: 36%; roberta alone 46%). Tried and rejected: training one
> model on both Codex and roberta labels (v3: better at English slang, but −7 points on our real posts).

### Authentication

Set `SENTIMENT_API_KEYS` (comma-separated) to require `X-API-Key: <key>` (or `Authorization: Bearer <key>`) on all data
endpoints. Without it the API is open, as before. `/health` and the dashboard page are always public. See
[deploy/oracle/README.md](deploy/oracle/README.md#api-keys-recommended-for-any-public-deployment).

### New response fields

| Field | Meaning |
| :--- | :--- |
| `tier` | `tfidf` (fast tier), `transformer` (multilingual) or `english_expert` (short English posts) |
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

