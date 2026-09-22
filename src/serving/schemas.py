from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class SentimentPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="Array of post texts to analyze", min_length=1)
    confidence_threshold: Optional[float] = Field(0.60, description="Confidence threshold below which fallback is required", ge=0.0, le=1.0)
    enable_fallback: Optional[bool] = Field(False, description="Whether to immediately route low-confidence posts to Jev API fallback")
    extract_aspects: Optional[bool] = Field(True, description="Whether to extract aspect-based sentiments (UI, speed, support, pricing)")

class SentimentPredictionItem(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    aspects: Dict[str, str] = Field(default_factory=dict)
    fallback_required: bool
    reason: Optional[str] = None
    cached: bool = False
    fallback_result: Optional[Dict[str, Any]] = None

class SentimentPredictResponse(BaseModel):
    results: List[SentimentPredictionItem]
    total_processed: int
    fallback_count: int
    cache_hits: int = 0
    batch_latency_ms: float
    model_version: str = "tfidf-context-absa-v3"

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
