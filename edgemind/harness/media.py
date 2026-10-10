import base64
import hashlib
import mimetypes
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path


IMAGE_LIMIT = 20 * 1024 * 1024
AUDIO_LIMIT = 40 * 1024 * 1024
_ASR_MODEL_CACHE = OrderedDict()
_ASR_MODEL_CACHE_LIMIT = 2
_ASR_MODEL_LOCK = threading.Lock()


def clear_asr_model_cache() -> None:
    with _ASR_MODEL_LOCK:
        _ASR_MODEL_CACHE.clear()


def _get_whisper_model(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    with _ASR_MODEL_LOCK:
        cached = _ASR_MODEL_CACHE.get(key)
        if cached is not None:
            _ASR_MODEL_CACHE.move_to_end(key)
            return cached, True, 0.0
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "语音转写组件未安装。请在项目 Python 环境运行 `uv sync --extra audio`，然后重启 EdgeMind。"
            ) from exc
        started = time.perf_counter()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        load_ms = round((time.perf_counter() - started) * 1000, 2)
        _ASR_MODEL_CACHE[key] = model
        _ASR_MODEL_CACHE.move_to_end(key)
        while len(_ASR_MODEL_CACHE) > _ASR_MODEL_CACHE_LIMIT:
            _ASR_MODEL_CACHE.popitem(last=False)
        return model, False, load_ms


class MediaProcessor:
    def __init__(self, whisper_model: str | None = None):
        self.whisper_model = whisper_model or os.getenv("EDGEMIND_WHISPER_MODEL", "small")

    @staticmethod
    def process_text(text: str) -> str:
        value = (text or "").strip()
        if not value:
            raise ValueError("请输入文字，或附上一张图片/一段音频。")
        return value

    def process_image(self, file_path: str | Path) -> tuple[dict, dict]:
        path = Path(file_path)
        if not path.is_file():
            raise ValueError("图片文件不存在或已过期，请重新上传。")
        data = path.read_bytes()
        if not data or len(data) > IMAGE_LIMIT:
            raise ValueError("图片为空或超过 20 MiB 限制。")
        mime = self._image_mime(data)
        if mime is None:
            raise ValueError("仅支持有效的 JPEG、PNG、GIF 或 WebP 图片。")
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        metadata = {"name": path.name, "mime": mime, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return {"type": "image_url", "image_url": {"url": data_url}}, metadata

    def process_audio(self, file_path: str | Path) -> tuple[str, dict]:
        path = Path(file_path)
        if not path.is_file():
            raise ValueError("音频文件不存在或已过期，请重新上传。")
        data = path.read_bytes()
        if not data or len(data) > AUDIO_LIMIT:
            raise ValueError("音频为空或超过 40 MiB 限制。")
        device = os.getenv("EDGEMIND_ASR_DEVICE", "cpu")
        compute_type = os.getenv("EDGEMIND_ASR_COMPUTE_TYPE", "int8")
        model, cache_hit, model_load_ms = _get_whisper_model(self.whisper_model, device, compute_type)
        transcribe_started = time.perf_counter()
        segments, info = model.transcribe(str(path), vad_filter=True)
        transcript = "".join(segment.text for segment in segments).strip()
        transcription_ms = round((time.perf_counter() - transcribe_started) * 1000, 2)
        if not transcript:
            raise ValueError("没有从音频中识别出语音内容。请检查录音是否包含清晰语音。")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        duration = getattr(info, "duration", None)
        duration = round(float(duration), 3) if duration is not None else None
        realtime_factor = round(transcription_ms / (duration * 1000), 4) if duration and duration > 0 else None
        return transcript, {
            "name": path.name, "mime": mime, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "asr_model": self.whisper_model,
            "language": getattr(info, "language", None), "model_cache_hit": cache_hit,
            "model_load_ms": model_load_ms, "transcription_ms": transcription_ms,
            "audio_duration_sec": duration, "realtime_factor": realtime_factor,
        }

    @staticmethod
    def _image_mime(data: bytes) -> str | None:
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None
