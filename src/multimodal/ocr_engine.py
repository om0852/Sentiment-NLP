import os
import sys
import shutil
import subprocess
import tempfile
from typing import Dict, Any, Optional

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

class OcrEngine:
    """
    Native, zero-external-binary OCR engine utilizing Windows.Media.Ocr on Windows,
    and fallback to Tesseract CLI on Linux / containerized environments.
    Includes automated image pre-processing (intelligent upscaling & autocontrast)
    to maximize OCR fidelity on low-resolution and mobile screenshots.
    """
    def __init__(self, script_path: Optional[str] = None):
        if script_path is None:
            script_path = os.path.join(os.path.dirname(__file__), "windows_ocr.ps1")
        self.script_path = os.path.abspath(script_path)

    def extract_text(self, image_input: Any) -> Dict[str, Any]:
        """
        Extracts text from image path (str) or raw image bytes (bytes).
        Returns metadata including text, dimensions, format, and method.
        """
        temp_file = None
        target_path = None
        img_info = {"width": 0, "height": 0, "format": "UNKNOWN", "mode": "UNKNOWN"}

        try:
            if isinstance(image_input, (bytes, bytearray)):
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file.write(image_input)
                temp_file.flush()
                temp_file.close()
                target_path = temp_file.name
            elif isinstance(image_input, str):
                target_path = os.path.abspath(image_input)
            else:
                return {
                    "text": "",
                    "metadata": img_info,
                    "has_text": False,
                    "method": "none",
                    "error": "Unsupported image input type"
                }

            if not os.path.exists(target_path):
                return {
                    "text": "",
                    "metadata": img_info,
                    "has_text": False,
                    "method": "none",
                    "error": f"Image file not found: {target_path}"
                }

            # 1. Image Pre-processing for Optimal OCR Fidelity
            ocr_image_path = target_path
            temp_upscaled = None

            if Image is not None:
                try:
                    with Image.open(target_path) as img:
                        img_info["width"], img_info["height"] = img.size
                        img_info["format"] = img.format or "UNKNOWN"
                        img_info["mode"] = img.mode

                        # If image is small or low-DPI (< 1200px width), upscale 2x and enhance contrast
                        if img.width < 1200 or img.height < 400:
                            new_w = max(img.width * 2, 800)
                            new_h = max(img.height * 2, int(800 * (img.height / max(1, img.width))))
                            enhanced = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
                            if ImageOps is not None:
                                try:
                                    enhanced = ImageOps.autocontrast(enhanced.convert("RGB"))
                                except Exception:
                                    pass

                            temp_upscaled = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            temp_upscaled_name = temp_upscaled.name
                            temp_upscaled.close()
                            enhanced.save(temp_upscaled_name, format="PNG")
                            ocr_image_path = temp_upscaled_name
                except Exception:
                    pass

            # 2. Invoke OCR Engine
            extracted_text = ""
            ocr_method = "none"

            if sys.platform == "win32" and os.path.exists(self.script_path):
                # Windows Native Media OCR
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", self.script_path,
                    "-ImagePath", ocr_image_path
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=15
                )
                if proc.returncode == 0:
                    extracted_text = proc.stdout.strip()
                    ocr_method = "windows_media_ocr"
            elif shutil.which("tesseract"):
                # Linux / Containerized Tesseract CLI if present
                proc = subprocess.run(
                    ["tesseract", ocr_image_path, "stdout"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=15
                )
                if proc.returncode == 0:
                    extracted_text = proc.stdout.strip()
                    ocr_method = "tesseract_ocr"
            else:
                ocr_method = "platform_ocr_unavailable"

            # Clean up temporary upscaled file
            if temp_upscaled and os.path.exists(temp_upscaled.name):
                try:
                    os.remove(temp_upscaled.name)
                except Exception:
                    pass

            has_text = len(extracted_text.strip()) > 0
            return {
                "text": extracted_text,
                "metadata": img_info,
                "has_text": has_text,
                "method": ocr_method,
                "error": None
            }

        except subprocess.TimeoutExpired:
            return {
                "text": "",
                "metadata": img_info,
                "has_text": False,
                "method": "ocr_timeout",
                "error": "OCR process timed out after 15s"
            }
        except Exception as e:
            return {
                "text": "",
                "metadata": img_info,
                "has_text": False,
                "method": "error",
                "error": str(e)
            }
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
