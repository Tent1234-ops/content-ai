from faster_whisper import WhisperModel
import importlib

print("Loading Whisper...", flush=True)

# Choose a larger model for better ASR accuracy. This uses more RAM/CPU or GPU.
# If a CUDA GPU is available, prefer float16 on GPU; otherwise use int8 on CPU.
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

model = WhisperModel(
    "medium",
    device=device,
    compute_type=compute_type,
)

def transcribe_with_meta(audio_path, language=None):
    transcribe_kwargs = {
        "beam_size": 5
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


def transcribe(audio_path):
    result = transcribe_with_meta(audio_path, language="th")
    return result["text"]
