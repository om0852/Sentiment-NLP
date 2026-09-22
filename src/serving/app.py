import os
import sys
import time
import json
import psutil
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.preprocessing.aspect_extractor import AspectExtractor
from src.models.tfidf_classifier import TfidfSentimentClassifier
from .schemas import (
    SentimentPredictRequest,
    SentimentPredictResponse,
    SentimentPredictionItem,
    HealthResponse,
    MetricsResponse
)
from .fallback import FallbackService
from .cache import SentimentLRUCache
from .active_learning import ActiveLearningQueue

# Global runtime state
APP_START_TIME = time.time()
pipeline: PreprocessingPipeline = None
context_analyzer: ContextAnalyzer = None
aspect_extractor: AspectExtractor = None
model: TfidfSentimentClassifier = None
fallback_service: FallbackService = None
cache: SentimentLRUCache = None
active_learning: ActiveLearningQueue = None

# Metrics counters
METRICS = {
    "total_requests": 0,
    "total_posts_analyzed": 0,
    "total_fallbacks_triggered": 0,
    "total_latency_ms": 0.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, context_analyzer, aspect_extractor, model, fallback_service, cache, active_learning
    print("Initializing sentiment engine runtime...", flush=True)
    
    # 1. Initialize preprocessing components
    pipeline = PreprocessingPipeline()
    context_analyzer = ContextAnalyzer()
    aspect_extractor = AspectExtractor()
    
    # 2. Initialize in-memory LRU cache and active learning queue
    cache = SentimentLRUCache(max_size=10000)
    active_learning = ActiveLearningQueue()
    
    # 3. Load trained model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.getenv("MODEL_PATH", os.path.join(project_root, "models", "sentiment_model.joblib"))
    
    model = TfidfSentimentClassifier()
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"Model successfully loaded from: {model_path}", flush=True)
    else:
        print(f"Warning: Model file not found at {model_path}. Please run train.py first.", flush=True)
        
    # 4. Initialize fallback service
    fallback_service = FallbackService()
    print("Sentiment engine runtime ready. Memory RSS:", round(psutil.Process().memory_info().rss / (1024 * 1024), 2), "MB", flush=True)
    
    yield
    print("Shutting down sentiment engine runtime...", flush=True)

app = FastAPI(
    title="Lightweight Social Vernacular Sentiment Engine",
    description="Ultra-lightweight sentiment analysis microservice optimized for Gen-Z slang, Hinglish, emojis, and negations within 300MB RAM / 0.1 vCPU.",
    version="1.2.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Sentiment Engine API</h1><p>Visit /docs for Swagger API documentation.</p>")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and model.is_trained,
        memory_rss_mb=round(rss_mb, 2),
        uptime_seconds=round(time.time() - APP_START_TIME, 2),
        version="1.2.0"
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    total_posts = METRICS["total_posts_analyzed"]
    fallback_pct = (METRICS["total_fallbacks_triggered"] / total_posts * 100) if total_posts > 0 else 0.0
    avg_latency = (METRICS["total_latency_ms"] / METRICS["total_requests"]) if METRICS["total_requests"] > 0 else 0.0
    
    return MetricsResponse(
        total_requests=METRICS["total_requests"],
        total_posts_analyzed=total_posts,
        total_fallbacks_triggered=METRICS["total_fallbacks_triggered"],
        fallback_rate_pct=round(fallback_pct, 2),
        avg_latency_ms=round(avg_latency, 3),
        cache_stats=cache.get_stats() if cache else {},
        active_learning_queued=active_learning.count_queued() if active_learning else 0
    )

@app.get("/api/v1/sentiment/active_learning")
async def get_active_learning_samples(limit: int = 50):
    if not active_learning or not os.path.exists(active_learning.log_path):
        return {"total_queued": 0, "samples": []}
    
    samples = []
    with open(active_learning.log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line.strip()))
                except Exception:
                    continue
    return {
        "total_queued": len(samples),
        "recent_samples": samples[-limit:]
    }

@app.post("/predict", response_model=SentimentPredictResponse)
@app.post("/api/v1/sentiment/predict", response_model=SentimentPredictResponse)
async def predict_sentiment(req: SentimentPredictRequest):
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded.")
        
    start_time = time.perf_counter()
    raw_texts = req.texts
    n_posts = len(raw_texts)
    
    result_items: List[SentimentPredictionItem] = []
    fallback_count = 0
    cache_hits = 0
    
    # 1. Identify which posts hit the LRU cache
    texts_to_process = []
    text_to_indices: Dict[str, List[int]] = {}
    
    # Pre-allocate results array
    resolved_results: List[Optional[SentimentPredictionItem]] = [None] * n_posts
    
    for i, text in enumerate(raw_texts):
        cached_item = cache.get(text) if cache else None
        if cached_item:
            cache_hits += 1
            # If aspects are requested but not cached, extract them
            if req.extract_aspects and not cached_item.get("aspects") and aspect_extractor:
                cached_item["aspects"] = aspect_extractor.extract_aspects(text)
            resolved_results[i] = SentimentPredictionItem(**cached_item)
        else:
            texts_to_process.append((i, text))
            
    # 2. Process uncached posts
    if texts_to_process:
        uncached_indices = [item[0] for item in texts_to_process]
        uncached_texts = [item[1] for item in texts_to_process]
        
        # Preprocess in batch
        processed_texts = []
        metas = []
        for t in uncached_texts:
            proc_t, meta = pipeline.process(t)
            processed_texts.append(proc_t)
            metas.append(meta)
            
        # Model predictions
        preds = model.predict_batch(processed_texts, confidence_threshold=req.confidence_threshold)
        
        # Analyze each uncached prediction
        for idx, orig_idx in enumerate(uncached_indices):
            raw_t = uncached_texts[idx]
            pred = preds[idx]
            meta = metas[idx]
            
            # Apply contextual reasoning
            final_label, conf, ctx_fallback, reason = context_analyzer.analyze(raw_t, pred["label"], pred["confidence"])
            
            needs_fallback = ctx_fallback or (conf < req.confidence_threshold) or meta.get("sarcasm_detected", False)
            if meta.get("sarcasm_detected", False):
                reason = "sarcasm_detected"
                
            fallback_res = None
            if needs_fallback:
                fallback_count += 1
                # Log to active learning queue for continuous feedback
                if active_learning:
                    active_learning.log_feedback(
                        text=raw_t,
                        initial_label=final_label,
                        confidence=conf,
                        reason=reason or "low_confidence"
                    )
                    
                if req.enable_fallback and fallback_service:
                    fallback_res = await fallback_service.fallback_predict(raw_t)
                    if fallback_res and fallback_res.get("resolved"):
                        final_label = fallback_res["label"]
                        conf = fallback_res["confidence"]
                        if active_learning:
                            active_learning.log_feedback(
                                text=raw_t,
                                initial_label=pred["label"],
                                confidence=conf,
                                reason="fallback_resolved",
                                resolved_label=final_label
                            )
                            
            # Aspect-based sentiment analysis
            aspects = {}
            if req.extract_aspects and aspect_extractor:
                aspects = aspect_extractor.extract_aspects(raw_t)
                
            pred_item = SentimentPredictionItem(
                text=raw_t,
                label=final_label,
                confidence=round(conf, 4),
                probabilities=pred["probabilities"],
                aspects=aspects,
                fallback_required=needs_fallback,
                reason=reason,
                cached=False,
                fallback_result=fallback_res
            )
            
            # Store into LRU cache
            if cache:
                cache.put(raw_t, pred_item.model_dump())
                
            resolved_results[orig_idx] = pred_item
            
    # Count any fallbacks from cached items if applicable
    for res in resolved_results:
        if res and res.cached and res.fallback_required:
            fallback_count += 1
            
    batch_latency = (time.perf_counter() - start_time) * 1000
    
    # Update metrics
    METRICS["total_requests"] += 1
    METRICS["total_posts_analyzed"] += n_posts
    METRICS["total_fallbacks_triggered"] += fallback_count
    METRICS["total_latency_ms"] += batch_latency
    
    return SentimentPredictResponse(
        results=[r for r in resolved_results if r is not None],
        total_processed=n_posts,
        fallback_count=fallback_count,
        cache_hits=cache_hits,
        batch_latency_ms=round(batch_latency, 3),
        model_version="tfidf-context-absa-v3"
    )
