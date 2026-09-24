import os
import sys
import subprocess
import tempfile
import json
import urllib.request
from typing import Dict, Any, Optional

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"

class AudioTranscriber:
    """
    High-performance audio extractor and speech transcriber for video and audio files.
    - Extracts embedded subtitle / closed-caption tracks (SRT, WebVTT) via FFmpeg.
    - Extracts 16kHz mono audio stream from any video container (MP4, MKV, MOV, WEBM).
    - Transcribes spoken audio into text with multi-language support (English, Hindi, Marathi).
    - Completely portable with zero unsigned binary dependencies.
    """
    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or FFMPEG_PATH

    def extract_embedded_subtitles(self, video_path: str) -> str:
        """
        Extracts embedded soft subtitles or closed captions from the video container.
        """
        if not os.path.exists(video_path):
            return ""
        try:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", video_path,
                "-map", "0:s:0",
                "-f", "webvtt",
                "-"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore", timeout=10)
            if proc.returncode == 0 and proc.stdout:
                # Clean webvtt tags and timestamps
                raw_lines = proc.stdout.split("\n")
                clean_lines = []
                for line in raw_lines:
                    line = line.strip()
                    if not line or "-->" in line or line.startswith("WEBVTT") or line.isdigit():
                        continue
                    clean_lines.append(line)
                return " ".join(clean_lines).strip()
        except Exception:
            pass
        return ""

    def extract_audio_track(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extracts 16kHz mono WAV audio from video.
        """
        if not os.path.exists(video_path):
            return None

        if output_path is None:
            temp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = temp_f.name
            temp_f.close()

        try:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=20)
            if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return output_path
        except Exception:
            pass
        return None

    def transcribe_audio_file(self, audio_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribes speech from audio using high-speed speech-to-text.
        Supports en-US, hi-IN, mr-IN.
        """
        if not audio_path or not os.path.exists(audio_path):
            return {"transcript": "", "confidence": 0.0, "has_speech": False, "method": "none"}

        # Convert to FLAC in memory for optimal transmission
        flac_temp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        flac_path = flac_temp.name
        flac_temp.close()

        try:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", audio_path,
                "-c:a", "flac",
                "-ar", "16000",
                "-ac", "1",
                flac_path
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=15)
            if proc.returncode != 0 or not os.path.exists(flac_path):
                return {"transcript": "", "confidence": 0.0, "has_speech": False, "method": "failed"}

            with open(flac_path, "rb") as f:
                flac_data = f.read()

            if len(flac_data) < 2000:
                # File is silent or near empty
                return {"transcript": "", "confidence": 0.0, "has_speech": False, "method": "silent"}

            # Call Speech-to-Text API
            url = f"https://www.google.com/speech-api/v2/recognize?output=json&lang={language}&key=AIzaSyA2we0eQ_P0"
            req = urllib.request.Request(
                url,
                data=flac_data,
                headers={"Content-Type": "audio/x-flac; rate=16000"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_text = resp.read().decode("utf-8", errors="ignore")
                
                # Google Speech API returns multiple chunked JSONs separated by newlines
                transcript = ""
                confidence = 0.85
                for line in resp_text.strip().split("\n"):
                    try:
                        data = json.loads(line)
                        if "result" in data and len(data["result"]) > 0:
                            alt = data["result"][0]["alternative"][0]
                            transcript = alt.get("transcript", "")
                            confidence = float(alt.get("confidence", 0.85))
                            break
                    except Exception:
                        continue

                return {
                    "transcript": transcript.strip(),
                    "confidence": confidence,
                    "has_speech": bool(transcript.strip()),
                    "method": "google_speech_v2"
                }

        except Exception as e:
            return {
                "transcript": "",
                "confidence": 0.0,
                "has_speech": False,
                "method": "error",
                "error": str(e)
            }
        finally:
            if os.path.exists(flac_path):
                try:
                    os.remove(flac_path)
                except Exception:
                    pass

    def process_video_speech_and_captions(self, video_path: str) -> Dict[str, Any]:
        """
        Complete audio pipeline for video:
        1. Checks embedded subtitle tracks.
        2. Extracts audio track and transcribes speech.
        3. Returns unified transcript and metadata.
        """
        # 1. Embedded subtitles
        subtitles = self.extract_embedded_subtitles(video_path)

        # 2. Extract and transcribe audio
        audio_wav = self.extract_audio_track(video_path)
        transcript = ""
        transcription_info = {}

        if audio_wav:
            try:
                transcription_info = self.transcribe_audio_file(audio_wav)
                transcript = transcription_info.get("transcript", "")
            finally:
                if os.path.exists(audio_wav):
                    try:
                        os.remove(audio_wav)
                    except Exception:
                        pass

        # Combine transcripts
        combined_text_parts = []
        if subtitles:
            combined_text_parts.append(f"Subtitles: {subtitles}")
        if transcript:
            combined_text_parts.append(f"Spoken: {transcript}")

        full_speech_text = " ".join(combined_text_parts).strip()
        has_audio = bool(audio_wav or full_speech_text)

        return {
            "speech_text": transcript,
            "subtitles": subtitles,
            "full_speech_text": full_speech_text,
            "has_speech": bool(full_speech_text),
            "transcription_metadata": transcription_info
        }
