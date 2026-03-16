from utils.audio import extract_audio
from models.speech_to_text import transcribe
from models.llm_qwen import fix_transcript, extract_keywords
from models.ner import extract_entities
from models.scene_detect import detect_scenes
from models.frame_extract import extract_frames
from models.blip_caption import caption_image

video = "videos/test.mp4"

print("Extract audio")
audio = extract_audio(video, "audio.wav")

print("Whisper")
raw_text = transcribe(audio)

print("Fix transcript")
clean_text = fix_transcript(raw_text)

print("Keywords")
keywords = extract_keywords(clean_text)

print("NER")
entities = extract_entities(clean_text)

print("Scene detect")
scenes = detect_scenes(video)

frames = extract_frames(video, scenes)

captions = []

for f in frames:

    captions.append(
        caption_image(f)
    )

print("Transcript:", clean_text)
print("Keywords:", keywords)
print("Entities:", entities)
print("Visual captions:", captions)