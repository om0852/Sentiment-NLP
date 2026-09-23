# ⚡ Ultra-Lightweight Pragmatic Sentiment Engine

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![RAM Footprint](https://img.shields.io/badge/RAM%20Budget-%3C300MB%20(Actual:%20125MB)-success.svg)]()
[![vCPU Budget](https://img.shields.io/badge/CPU%20Budget-0.1%20vCPU-success.svg)]()
[![Cost](https://img.shields.io/badge/Cloud%20Cost-%240.00%2Fmo%20(Free%20Tier)-brightgreen.svg)]()
[![Benchmark Hard Sets](https://img.shields.io/badge/Hard%20Benchmarks-100%2F100%20(100%25)-brightgreen.svg)]()
[![300k Evaluation](https://img.shields.io/badge/Production%20300k%20Stream-923%20posts%2Fsec-blue.svg)]()
[![Live Deployment](https://img.shields.io/badge/Railway-Live%20Production-success.svg)](https://sentiment-nlp-production.up.railway.app/)

A production-grade, ultra-lightweight sentiment analysis microservice specialized in modern social vernacular (**Gen-Z slang, multi-modal emojis, complex negations, contrastive clauses, Hinglish, cultural disaster tropes, and thread-context sarcasm**).

Operating strictly within **$\le 300\text{ MB}$ RAM** and **$\le 0.1\text{ vCPU}$**, processing **$100,000\text{ to }1,000,000+\text{ posts/day}$** at **$\$0.00\text{ cloud cost}$**.

---

## 🌟 Live Demo & Interactive Dashboard

- **Live Microservice URL:** [`https://sentiment-nlp-production.up.railway.app/`](https://sentiment-nlp-production.up.railway.app/)
- **Interactive Glassmorphic Playground:** Served directly at root `GET /`
- **Health Check Endpoint:** `GET /health`
- **Real-Time Telemetry & Cache Stats:** `GET /metrics`

---

## 📊 Key Achievements & Performance Metrics

| Metric | Budget / Target | Measured In Production | Safety Headroom |
| :--- | :--- | :--- | :--- |
| **Hard Benchmark Accuracy** | $\ge 85.0\%$ | **100.0% (100 / 100 passed)** | Outperformed 70B LLMs |
| **1,000 Boundary Stress Test**| $\ge 90.0\%$ | **100.0% (1,000 / 1,000 passed)** | 10 complex NLP failure modes |
| **300k Real MongoDB Stream**  | 300,000 posts | **923 posts / sec (325.0s total)** | 158.7k exact / 141.2k refined |
| **Statistical Latency** | $< 2.0\text{ ms}$ | **0.058 - 0.087 ms** | **$23\times$ faster** |
| **End-to-End Latency** | $< 10.0\text{ ms}$ | **1.08 - 1.56 ms** | **$6.4\times$ faster** |
| **LRU Cache Hit Latency** | $< 0.5\text{ ms}$ | **0.069 ms** | **$28.4\times$ speedup** |
| **Throughput (1 Core)** | $> 200\text{ posts/s}$ | **1,683 posts / second** | **$8.4\times$ higher** |
| **Throughput (0.1 vCPU)** | $\sim 11.6\text{ posts/s}$ ($10\text{L/day}$) | **168.3 posts / second** | **$14.5\times$ daily target** |
| **Runtime Memory (RSS)** | $\le 300\text{ MB}$ | **125.2 MB (local) / 148.7 MB (Railway)** | **$2.0\times$ under budget** |
| **Model Disk Footprint** | $\le 25\text{ MB}$ | **0.48 MB** | **$52.1\times$ smaller** |
| **Operating Cost** | Budget $\le \$0.00$ | **$0.00 / month** | 100% Free-Tier |

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

# 4. Run automated test suite
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

