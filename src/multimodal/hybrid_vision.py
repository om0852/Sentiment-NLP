import os
import io
from typing import Dict, Any, Optional
from PIL import Image, ImageStat

class HybridVisionAnalyzer:
    """
    Hybrid Vision Analyzer supporting both offline local heuristics and pluggable external Vision AI.
    Analyzes visual tone, color palette, dark mode detection, and payment UI brand signatures.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VISION_AI_API_KEY")
        self.enabled_cloud = bool(self.api_key)

    def analyze_visual_scene(self, image_input: Any) -> Dict[str, Any]:
        """
        Extracts visual cues (dark mode, average brightness, payment app color signatures, visual tone).
        """
        try:
            if isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, str) and os.path.exists(image_input):
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return {
                    "visual_tone": "neutral",
                    "is_dark_mode": False,
                    "brightness": 128.0,
                    "is_payment_ui": False,
                    "detected_gateway": None
                }

            if img.mode != "RGB":
                img = img.convert("RGB")

            # Calculate brightness and color statistics
            stat = ImageStat.Stat(img)
            r_mean, g_mean, b_mean = stat.mean[0], stat.mean[1], stat.mean[2]
            perceived_brightness = round(0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean, 1)

            is_dark_mode = perceived_brightness < 80.0
            
            if perceived_brightness > 180.0:
                visual_tone = "bright / vibrant"
            elif perceived_brightness < 60.0:
                visual_tone = "dark / moody"
            else:
                visual_tone = "balanced"

            # 2. Detect Payment App Brand & UI Signatures
            # PhonePe: Signature Purple (#5f259f)
            # Paytm: Navy/Cyan (#002e6e / #00b9f5)
            # GPay: Multi-color / Google Blue (#1a73e8)
            pixels = list(img.getdata())
            purple_count = 0
            blue_count = 0
            total_px = max(1, len(pixels))

            for p in pixels:
                r, g, b = p[0], p[1], p[2]
                if b > 90 and r > 60 and g < 60 and (b - g) > 40:
                    purple_count += 1
                elif b > 140 and r < 70 and g > 50:
                    blue_count += 1

            purple_ratio = purple_count / total_px
            blue_ratio = blue_count / total_px

            is_payment_ui = False
            detected_gateway = None

            if purple_ratio >= 0.04:
                is_payment_ui = True
                detected_gateway = "PhonePe / UPI Theme"
            elif blue_ratio >= 0.08:
                is_payment_ui = True
                detected_gateway = "Paytm / Banking Gateway"

            return {
                "visual_tone": visual_tone,
                "is_dark_mode": is_dark_mode,
                "brightness": perceived_brightness,
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 1.0,
                "is_payment_ui": is_payment_ui,
                "detected_gateway": detected_gateway,
                "cloud_ai_used": False
            }

        except Exception as e:
            return {
                "visual_tone": "unknown",
                "is_dark_mode": False,
                "brightness": 128.0,
                "is_payment_ui": False,
                "detected_gateway": None,
                "error": str(e)
            }
