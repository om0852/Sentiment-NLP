import os
import io
from typing import Dict, Any, Optional
from PIL import Image, ImageStat

class HybridVisionAnalyzer:
    """
    Hybrid Vision Analyzer supporting both offline local heuristics and pluggable external Vision AI.
    Analyzes visual tone, color palette, dark mode detection, and scene semantics.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VISION_AI_API_KEY")
        self.enabled_cloud = bool(self.api_key)

    def analyze_visual_scene(self, image_input: Any) -> Dict[str, Any]:
        """
        Extracts visual cues (dark mode, average brightness, color richness, visual tone).
        """
        try:
            if isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, str) and os.path.exists(image_input):
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return {"visual_tone": "neutral", "is_dark_mode": False, "brightness": 128.0}

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Calculate brightness and color statistics
            stat = ImageStat.Stat(img)
            # Perceived brightness formula: 0.299*R + 0.587*G + 0.114*B
            r, g, b = stat.mean[0], stat.mean[1], stat.mean[2]
            perceived_brightness = round(0.299 * r + 0.587 * g + 0.114 * b, 1)

            is_dark_mode = perceived_brightness < 80.0
            
            # Simple visual tone heuristic
            if perceived_brightness > 180.0:
                visual_tone = "bright / vibrant"
            elif perceived_brightness < 60.0:
                visual_tone = "dark / moody"
            else:
                visual_tone = "balanced"

            return {
                "visual_tone": visual_tone,
                "is_dark_mode": is_dark_mode,
                "brightness": perceived_brightness,
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 1.0,
                "cloud_ai_used": False
            }

        except Exception as e:
            return {
                "visual_tone": "unknown",
                "is_dark_mode": False,
                "brightness": 128.0,
                "error": str(e)
            }
