import os
import sys
import tempfile
import cv2
from typing import Dict, Any, List, Optional
from PIL import Image
from src.multimodal.ocr_engine import OcrEngine
from src.multimodal.audio_transcriber import AudioTranscriber

class VideoProcessor:
    """
    Comprehensive Video Intelligence Processor.
    1. Extracts keyframes across video duration & runs OCR on on-screen text, titles & burned-in captions.
    2. Extracts embedded subtitles & closed captions (SRT / WebVTT).
    3. Extracts audio track & transcribes spoken speech into text.
    Combines visual text + spoken audio into a unified semantic content stream.
    """
    def __init__(self, ocr_engine: Optional[OcrEngine] = None, audio_transcriber: Optional[AudioTranscriber] = None):
        self.ocr_engine = ocr_engine or OcrEngine()
        self.audio_transcriber = audio_transcriber or AudioTranscriber()

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
                    "spoken_text": "",
                    "ocr_text": "",
                    "subtitles": "",
                    "metadata": {},
                    "keyframes": [],
                    "success": False,
                    "error": "Unsupported video input type"
                }

            if not os.path.exists(target_path):
                return {
                    "text": "",
                    "spoken_text": "",
                    "ocr_text": "",
                    "subtitles": "",
                    "metadata": {},
                    "keyframes": [],
                    "success": False,
                    "error": f"Video file not found: {target_path}"
                }

            # --- A. VIDEO METADATA & KEYFRAME EXTRACTION ---
            cap = cv2.VideoCapture(target_path)
            if not cap.isOpened():
                return {
                    "text": "",
                    "spoken_text": "",
                    "ocr_text": "",
                    "subtitles": "",
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

            ocr_extracted_texts = []
            keyframe_info = []

            if total_frames > 0:
                step = total_frames / (max_frames + 1)
                frame_indices = [int(step * (i + 1)) for i in range(max_frames)]

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    timestamp_sec = round(idx / fps, 2)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
                        temp_img_path = f_img.name

                    try:
                        pil_img.save(temp_img_path)
                        ocr_res = self.ocr_engine.extract_text(temp_img_path)
                        frame_text = ocr_res.get("text", "").strip()
                        if frame_text and frame_text not in ocr_extracted_texts:
                            ocr_extracted_texts.append(frame_text)
                        keyframe_info.append({
                            "frame_index": idx,
                            "timestamp_seconds": timestamp_sec,
                            "text": frame_text
                        })
                    finally:
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)

            cap.release()
            ocr_text = " ".join(ocr_extracted_texts).strip()

            # --- B. AUDIO & SPEECH TRANSCRIPTION ---
            audio_res = self.audio_transcriber.process_video_speech_and_captions(target_path)
            spoken_text = audio_res.get("speech_text", "")
            subtitles = audio_res.get("subtitles", "")

            # Combine all text channels into unified semantic stream
            combined_streams = []
            if ocr_text:
                combined_streams.append(ocr_text)
            if spoken_text:
                combined_streams.append(spoken_text)
            if subtitles:
                combined_streams.append(subtitles)

            full_video_text = " ".join(combined_streams).strip()

            metadata["has_audio_track"] = bool(spoken_text or audio_res.get("has_speech"))
            metadata["has_spoken_speech"] = bool(spoken_text)
            metadata["has_embedded_subtitles"] = bool(subtitles)
            metadata["spoken_word_count"] = len(spoken_text.split()) if spoken_text else 0

            return {
                "text": full_video_text,
                "spoken_text": spoken_text,
                "ocr_text": ocr_text,
                "subtitles": subtitles,
                "metadata": metadata,
                "keyframes": keyframe_info,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "text": "",
                "spoken_text": "",
                "ocr_text": "",
                "subtitles": "",
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
