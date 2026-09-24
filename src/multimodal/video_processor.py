import os
import sys
import tempfile
import cv2
from typing import Dict, Any, List, Optional
from PIL import Image
from src.multimodal.ocr_engine import OcrEngine

class VideoProcessor:
    """
    High-performance video analysis module using OpenCV.
    Samples keyframes across video duration, extracts on-screen text, subtitles, and captions via OCR.
    """
    def __init__(self, ocr_engine: Optional[OcrEngine] = None):
        self.ocr_engine = ocr_engine or OcrEngine()

    def process_video(self, video_input: Any, max_frames: int = 5) -> Dict[str, Any]:
        temp_file = None
        target_path = None

        try:
            if isinstance(video_input, (bytes, bytearray)):
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.write(video_input)
                temp_file.flush()
                temp_file.close()
                target_path = temp_file.name
            elif isinstance(video_input, str):
                target_path = os.path.abspath(video_input)
            else:
                return {
                    "text": "",
                    "metadata": {},
                    "keyframes": [],
                    "success": False,
                    "error": "Unsupported video input type"
                }

            if not os.path.exists(target_path):
                return {
                    "text": "",
                    "metadata": {},
                    "keyframes": [],
                    "success": False,
                    "error": f"Video file not found: {target_path}"
                }

            cap = cv2.VideoCapture(target_path)
            if not cap.isOpened():
                return {
                    "text": "",
                    "metadata": {},
                    "keyframes": [],
                    "success": False,
                    "error": "Failed to open video stream"
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

            metadata = {
                "duration_seconds": duration_sec,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}"
            }

            if total_frames <= 0:
                cap.release()
                return {"text": "", "metadata": metadata, "keyframes": [], "success": True, "error": None}

            # Select frame indices evenly across the video duration
            frame_indices = []
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames / (max_frames + 1)
                frame_indices = [int(step * (i + 1)) for i in range(max_frames)]

            extracted_texts = []
            keyframe_info = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                timestamp_sec = round(idx / fps, 2)
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Save temp frame for OCR
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
                    temp_img_path = f_img.name
                
                try:
                    pil_img.save(temp_img_path)
                    ocr_res = self.ocr_engine.extract_text(temp_img_path)
                    frame_text = ocr_res.get("text", "").strip()
                    if frame_text and frame_text not in extracted_texts:
                        extracted_texts.append(frame_text)
                    keyframe_info.append({
                        "frame_index": idx,
                        "timestamp_seconds": timestamp_sec,
                        "text": frame_text
                    })
                finally:
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

            cap.release()
            full_video_text = " ".join(extracted_texts).strip()

            return {
                "text": full_video_text,
                "metadata": metadata,
                "keyframes": keyframe_info,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "text": "",
                "metadata": {},
                "keyframes": [],
                "success": False,
                "error": str(e)
            }
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
