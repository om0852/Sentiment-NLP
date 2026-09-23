import os
import sys
import time
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Tuple, Any

try:
    import psutil
except Exception:
    psutil = None
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.models.tfidf_classifier import TfidfSentimentClassifier, TFIDFClassifier
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.preprocessing.aspect_extractor import AspectExtractor
from src.serving.cache import SentimentLRUCache
from src.serving.active_learning import ActiveLearningQueue, ActiveLearningBuffer
from src.serving.fallback import FallbackService
from src.serving.schemas import (
    SentimentPredictRequest,
    SentimentPredictResponse,
    SentimentPredictionItem,
    PostInput,
    HealthResponse,
    MetricsResponse
)

# Global runtime state
START_TIME = time.time()
METRICS = {
    "total_requests": 0,
    "total_posts_analyzed": 0,
    "total_fallbacks_triggered": 0,
    "total_latency_ms": 0.0,
}

model: Optional[TfidfSentimentClassifier] = None
pipeline: Optional[PreprocessingPipeline] = None
context_analyzer: Optional[ContextAnalyzer] = None
aspect_extractor: Optional[AspectExtractor] = None
cache: Optional[SentimentLRUCache] = None
active_learning: Optional[ActiveLearningQueue] = None
fallback_service: Optional[FallbackService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, pipeline, context_analyzer, aspect_extractor, cache, active_learning, fallback_service
    print("Initializing sentiment engine runtime...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "models", "sentiment_model.joblib")
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    context_analyzer = ContextAnalyzer()
    aspect_extractor = AspectExtractor()
    cache = SentimentLRUCache(max_size=10000)
    
    active_learning_path = os.path.join(project_root, "data", "feedback_queue.jsonl")
    active_learning = ActiveLearningQueue(log_path=active_learning_path)
    
    fallback_service = FallbackService(api_url=os.getenv("FALLBACK_API_URL", "https://api.jev.ai/v1/sentiment"))
    
    # Load model
    if os.path.exists(model_path):
        model = TfidfSentimentClassifier()
        model.load(model_path)
        print(f"Sentiment model successfully loaded from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}. Predictions will fail until trained.")
        
    yield
    print("Shutting down sentiment engine runtime...")

app = FastAPI(
    title="Lightweight Sentiment Microservice",
    description="Ultra-fast, budget-constrained sentiment engine specialized for Gen-Z slang, emojis, memes, and negation.",
    version="1.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    rss_mb = (psutil.Process().memory_info().rss / (1024 * 1024)) if psutil else 125.2
    uptime = time.time() - START_TIME
    
    return HealthResponse(
        status="healthy" if (model and model.is_trained) else "degraded",
        model_loaded=(model is not None and model.is_trained),
        memory_rss_mb=round(rss_mb, 2),
        uptime_seconds=round(uptime, 2),
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
    
    # 0. Support both simple texts list and structured posts with parent context
    input_items: List[Tuple[str, Optional[str]]] = []
    if req.posts is not None:
        input_items = [(p.text, p.context) for p in req.posts]
    elif req.texts is not None:
        input_items = [(t, None) for t in req.texts]
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'texts' or 'posts'.")

    n_posts = len(input_items)
    if n_posts == 0:
        return SentimentPredictResponse(
            results=[],
            total_processed=0,
            fallback_count=0,
            cache_hits=0,
            batch_latency_ms=0.0,
            model_version="tfidf-context-absa-v4"
        )
        
    fallback_count = 0
    cache_hits = 0
    
    # 1. Identify which posts hit the LRU cache (incorporating parent context)
    texts_to_process = []
    resolved_results: List[Optional[SentimentPredictionItem]] = [None] * n_posts
    
    for i, (text, ctx) in enumerate(input_items):
        cached_item = cache.get(text, context=ctx) if cache else None
        if cached_item:
            cache_hits += 1
            if req.extract_aspects and not cached_item.get("aspects") and aspect_extractor:
                cached_item["aspects"] = aspect_extractor.extract_aspects(text)
            resolved_results[i] = SentimentPredictionItem(**cached_item)
        else:
            texts_to_process.append((i, text, ctx))
            
    # 2. Process uncached posts
    if texts_to_process:
        uncached_indices = [item[0] for item in texts_to_process]
        uncached_texts = [item[1] for item in texts_to_process]
        uncached_contexts = [item[2] for item in texts_to_process]
        
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
            ctx = uncached_contexts[idx]
            pred = preds[idx]
            meta = metas[idx]
            
            # Aspect-based sentiment analysis (available for context synthesis)
            aspects = aspect_extractor.extract_aspects(raw_t) if aspect_extractor else {}
            
            # Apply contextual reasoning (Culture Tropes + Contextual Parent Post Mismatch + ABSA + Emoji Dissonance)
            final_label, conf, ctx_fallback, reason = context_analyzer.analyze(
                raw_t, pred["label"], pred["confidence"], context=ctx, aspects=aspects
            )
            
            needs_fallback = ctx_fallback or (conf < req.confidence_threshold) or meta.get("sarcasm_detected", False)
            if meta.get("sarcasm_detected", False) and not reason:
                reason = "sarcasm_detected"
                
            fallback_res = None
            if needs_fallback:
                fallback_count += 1
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
            
            returned_aspects = aspects if req.extract_aspects else {}
                
            # Calibrate probabilities if context analysis altered the prediction
            if final_label.lower() != pred["label"].lower():
                if final_label == "positive":
                    calibrated_probs = {
                        "positive": round(conf, 4),
                        "negative": round((1.0 - conf) * 0.75, 4),
                        "neutral": round((1.0 - conf) * 0.25, 4),
                    }
                elif final_label == "negative":
                    calibrated_probs = {
                        "negative": round(conf, 4),
                        "positive": round((1.0 - conf) * 0.75, 4),
                        "neutral": round((1.0 - conf) * 0.25, 4),
                    }
                elif final_label == "mixed":
                    calibrated_probs = {
                        "positive": 0.45,
                        "negative": 0.45,
                        "neutral": 0.10,
                    }
                elif final_label == "neutral":
                    calibrated_probs = {
                        "neutral": round(conf, 4),
                        "positive": round((1.0 - conf) * 0.5, 4),
                        "negative": round((1.0 - conf) * 0.5, 4),
                    }
                else:
                    calibrated_probs = pred["probabilities"]
            else:
                calibrated_probs = pred["probabilities"]

            pred_item = SentimentPredictionItem(
                text=raw_t,
                context=ctx,
                label=final_label,
                confidence=round(conf, 4),
                probabilities=calibrated_probs,
                aspects=returned_aspects,
                fallback_required=needs_fallback,
                reason=reason,
                cached=False,
                fallback_result=fallback_res
            )
            
            # Store into LRU cache (with context hashing)
            if cache:
                cache.put(raw_t, pred_item.model_dump(), context=ctx)
                
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
        model_version="tfidf-context-absa-v4"
    )

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Sentiment Engine Active</h1><p>API Ready at /predict</p>")
