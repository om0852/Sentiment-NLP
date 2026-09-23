from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class PostInput(BaseModel):
    text: str = Field(..., description="Post text to analyze")
    context: Optional[str] = Field(None, description="Optional parent post, topic title, or incident context")

class SentimentPredictRequest(BaseModel):
    texts: Optional[List[str]] = Field(None, description="Array of post texts to analyze (simple mode)")
    posts: Optional[List[PostInput]] = Field(None, description="Array of posts with optional parent/thread context (contextual mode)")
    confidence_threshold: Optional[float] = Field(0.60, description="Confidence threshold below which fallback is required", ge=0.0, le=1.0)
    enable_fallback: Optional[bool] = Field(False, description="Whether to immediately route low-confidence posts to Jev API fallback")
    extract_aspects: Optional[bool] = Field(True, description="Whether to extract aspect-based sentiments (UI, speed, support, pricing)")

class SentimentPredictionItem(BaseModel):
    text: str
    context: Optional[str] = None
    label: str
    confidence: float
    probabilities: Dict[str, float]
    aspects: Dict[str, str] = Field(default_factory=dict)
    fallback_required: bool
    reason: Optional[str] = None
    cached: bool = False
    fallback_result: Optional[Dict[str, Any]] = None
    is_performance_issue: bool = False
    is_risk_complaint: bool = False
    is_mixed: bool = False

class SentimentPredictResponse(BaseModel):
    results: List[SentimentPredictionItem]
    total_processed: int
    fallback_count: int
    cache_hits: int = 0
    batch_latency_ms: float
    model_version: str = "tfidf-context-absa-v4"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_rss_mb: float
    uptime_seconds: float
    version: str

class MetricsResponse(BaseModel):
    total_requests: int
    total_posts_analyzed: int
    total_fallbacks_triggered: int
    fallback_rate_pct: float
    avg_latency_ms: float
    cache_stats: Dict[str, Any]
    active_learning_queued: int
