import os
import sys
import time
from typing import Dict, Any, Optional

from src.models import PureSentimentClassifier
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.context_analyzer import ContextAnalyzer
from src.preprocessing.aspect_extractor import AspectExtractor
from src.multimodal.ocr_engine import OcrEngine
from src.multimodal.document_reader import DocumentReader
from src.multimodal.video_processor import VideoProcessor
from src.multimodal.categorizer import DomainCategorizer
from src.multimodal.tagger import SemanticTagger
from src.multimodal.hybrid_vision import HybridVisionAnalyzer

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "flv"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "doc", "csv", "json", "txt", "md", "log"}

class MultimodalPipeline:
    """
    Unified Multimodal Intelligence Pipeline.
    Analyzes images, videos, and documents to extract Sentiment, Domain Category, Semantic Tags, and Text.
    """
    def __init__(self, model_path: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if model_path is None:
            model_path = os.path.join(project_root, "models", "sentiment_model.json.gz")

        # 1. Core NLP & Sentiment Engine
        self.text_pipeline = PreprocessingPipeline()
        self.context_analyzer = ContextAnalyzer()
        self.aspect_extractor = AspectExtractor()
        self.sentiment_model = PureSentimentClassifier()
        if os.path.exists(model_path):
            self.sentiment_model.load(model_path)

        # 2. Multimodal Extraction & Analysis Engines
        self.ocr_engine = OcrEngine()
        self.doc_reader = DocumentReader()
        self.video_processor = VideoProcessor(self.ocr_engine)
        self.categorizer = DomainCategorizer()
        self.tagger = SemanticTagger()
        self.vision_analyzer = HybridVisionAnalyzer()

    def detect_media_type(self, filename: str, mime_type: str = "") -> str:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext in IMAGE_EXTENSIONS or mime_type.startswith("image/"):
            return "image"
        elif ext in VIDEO_EXTENSIONS or mime_type.startswith("video/"):
            return "video"
        elif ext in DOCUMENT_EXTENSIONS or mime_type.startswith("application/pdf") or mime_type.startswith("text/"):
            return "document"
        return "document"

    def analyze(self, file_input: Any, filename: str = "", mime_type: str = "") -> Dict[str, Any]:
        start_time = time.perf_counter()
        media_type = self.detect_media_type(filename, mime_type)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")

        extracted_text = ""
        media_metadata = {"media_type": media_type}
        has_text = False

        # --- 1. MEDIA EXTRACTION ---
        if media_type == "image":
            ocr_res = self.ocr_engine.extract_text(file_input)
            extracted_text = ocr_res.get("text", "").strip()
            has_text = bool(extracted_text)
            media_metadata.update(ocr_res.get("metadata", {}))
            
            # Extract visual cues & payment gateway brand signatures
            visual_cues = self.vision_analyzer.analyze_visual_scene(file_input)
            media_metadata.update(visual_cues)

        elif media_type == "video":
            vid_res = self.video_processor.process_video(file_input)
            extracted_text = vid_res.get("text", "").strip()
            has_text = bool(extracted_text)
            media_metadata.update(vid_res.get("metadata", {}))
            media_metadata["keyframes_sampled"] = len(vid_res.get("keyframes", []))

        elif media_type in ("document", "file"):
            if isinstance(file_input, (bytes, bytearray)):
                doc_res = self.doc_reader.extract_text_from_bytes(file_input, filename=filename, file_format=ext)
            elif isinstance(file_input, str):
                if os.path.exists(file_input):
                    doc_res = self.doc_reader.extract_text_from_file(file_input)
                else:
                    doc_res = {"text": file_input, "page_count": 1, "metadata": {}, "success": True}
            else:
                doc_res = {"text": str(file_input), "page_count": 1, "metadata": {}, "success": True}
            
            extracted_text = doc_res.get("text", "").strip()
            has_text = bool(extracted_text)
            media_metadata["page_count"] = doc_res.get("page_count", 1)
            media_metadata.update(doc_res.get("metadata", {}))

        # --- 2. SENTIMENT INFERENCE ---
        if has_text:
            cleaned_text, pre_meta = self.text_pipeline.process(extracted_text)
            model_res = self.sentiment_model.predict_one(cleaned_text)
            aspects = self.aspect_extractor.extract_aspects(extracted_text)
            
            final_label, conf, overridden, reason = self.context_analyzer.analyze(
                extracted_text,
                model_res["label"],
                model_res["confidence"],
                aspects=aspects
            )
            
            probabilities = model_res.get("probabilities", {final_label: conf})
            if overridden:
                probabilities = {
                    "negative": 0.90 if final_label == "negative" else 0.05,
                    "neutral": 0.90 if final_label == "neutral" else 0.05,
                    "positive": 0.90 if final_label == "positive" else 0.05
                }
                probabilities[final_label] = conf
        else:
            final_label = "neutral"
            conf = 0.50
            probabilities = {"negative": 0.20, "neutral": 0.60, "positive": 0.20}
            reason = "visual_default_no_text"
            aspects = {}

        sentiment_payload = {
            "label": final_label,
            "confidence": round(conf, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "reason": reason,
            "aspects": aspects
        }

        # --- 3. DOMAIN CATEGORIZATION ---
        cat_result = self.categorizer.classify(extracted_text, metadata=media_metadata)

        # --- 4. SEMANTIC TAGGING ---
        tags = self.tagger.extract_tags(
            extracted_text,
            category=cat_result["category"],
            sentiment_label=final_label,
            metadata=media_metadata
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "filename": filename or f"unnamed.{ext or media_type}",
            "media_type": media_type,
            "file_format": ext or media_type,
            "sentiment": sentiment_payload,
            "category": cat_result["category"],
            "subcategory": cat_result["subcategory"],
            "category_confidence": cat_result["confidence"],
            "tags": tags,
            "extracted_text": extracted_text,
            "has_text": has_text,
            "metadata": media_metadata,
            "latency_ms": latency_ms
        }
