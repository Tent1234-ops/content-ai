import importlib
import os
from typing import Dict

# Lazy model manager that loads WhisperModel per requested size

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# runtime config
from app.runtime import get as runtime_get

_MODEL_STORE: Dict[str, Dict] = {}


def _detect_device_compute():
    device = "cpu"
    compute_type = "int8"
    if importlib.util.find_spec("torch"):
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass
    return device, compute_type


class ModelManager:
    @staticmethod
    def get_model(size: str):
        """Return a WhisperModel instance for given size; load lazily and cache."""
        size = (size or "small").lower()
        if size in _MODEL_STORE:
            return _MODEL_STORE[size]["model"]

        if WhisperModel is None:
            raise RuntimeError("faster_whisper not installed or failed to import")

        device, compute_type = _detect_device_compute()
        # smaller sizes can use smaller beam and int8 for CPU
        compute = compute_type
        model_kwargs = {
            "device": device,
            "compute_type": compute,
        }
        if os.getenv("ASR_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}:
            model_kwargs["local_files_only"] = True

        try:
            model = WhisperModel(size, **model_kwargs)
        except TypeError:
            model = WhisperModel(size, device=device, compute_type=compute)
        except Exception:
            # fallback to CPU int8
            fallback_kwargs = {"device": "cpu", "compute_type": "int8"}
            if "local_files_only" in model_kwargs:
                fallback_kwargs["local_files_only"] = True
            model = WhisperModel(size, **fallback_kwargs)

        _MODEL_STORE[size] = {
            "model": model,
            "device": device,
            "compute_type": compute,
        }
        return model


def transcribe_with_meta(audio_path, language=None, model_size: str = None):
    # Decide model size from runtime if not provided
    model_size = model_size or runtime_get("asr_model") or "tiny"
    model = ModelManager.get_model(model_size)

    transcribe_kwargs = {
        "beam_size": 5 if model_size in {"medium", "large"} else 1,
        "vad_filter": True,
    }
    if language:
        transcribe_kwargs["language"] = language

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    text_parts = []
    total_no_speech_prob = 0.0
    segment_count = 0
    segments_out = []

    for seg in segments:
        seg_text = (seg.text or "").strip()
        seg_no_sp = getattr(seg, "no_speech_prob", None)
        seg_start = getattr(seg, "start", None)
        seg_end = getattr(seg, "end", None)
        segments_out.append({
            "text": seg_text,
            "no_speech_prob": float(seg_no_sp) if seg_no_sp is not None else None,
            "start": float(seg_start) if seg_start is not None else None,
            "end": float(seg_end) if seg_end is not None else None,
        })
        if seg_text:
            text_parts.append(seg_text)
        if seg_no_sp is not None:
            total_no_speech_prob += float(seg_no_sp)
            segment_count += 1

    avg_no_speech_prob = (total_no_speech_prob / segment_count) if segment_count else None

    return {
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segment_count": len(text_parts),
        "avg_no_speech_prob": avg_no_speech_prob,
        "segments": segments_out,
    }


def transcribe(audio_path, model_size: str = None):
    result = transcribe_with_meta(audio_path, language="th", model_size=model_size)
    return result["text"]
