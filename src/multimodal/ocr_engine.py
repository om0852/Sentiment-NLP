import os
import sys
import shutil
import subprocess
import tempfile
from typing import Dict, Any, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

class OcrEngine:
    """
    Native, zero-external-binary OCR engine utilizing Windows.Media.Ocr on Windows,
    and fallback to Tesseract CLI on Linux / containerized environments.
    Extracts text from images, memes, receipts, screenshots, and infographics.
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

            # 1. Read PIL Metadata if Pillow is available
            if Image is not None:
                try:
                    with Image.open(target_path) as img:
                        img_info["width"], img_info["height"] = img.size
                        img_info["format"] = img.format or "UNKNOWN"
                        img_info["mode"] = img.mode
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
                    "-ImagePath", target_path
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
                    ["tesseract", target_path, "stdout"],
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
