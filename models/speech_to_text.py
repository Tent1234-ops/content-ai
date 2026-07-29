from faster_whisper import WhisperModel

print("Loading Whisper...", flush=True)

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
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

    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            text_parts.append(seg_text)
        if getattr(seg, "no_speech_prob", None) is not None:
            total_no_speech_prob += float(seg.no_speech_prob)
            segment_count += 1

    avg_no_speech_prob = (total_no_speech_prob / segment_count) if segment_count else None

    return {
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segment_count": len(text_parts),
        "avg_no_speech_prob": avg_no_speech_prob,
    }


def transcribe(audio_path):
    result = transcribe_with_meta(audio_path, language="th")
    return result["text"]
