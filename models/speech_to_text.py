from faster_whisper import WhisperModel

print("Loading Whisper...")

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

def transcribe(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="th"
    )

    text = ""

    for seg in segments:
        text += seg.text + " "

    return text