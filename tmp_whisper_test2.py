from utils.audio import extract_audio
from models.speech_to_text import transcribe_with_meta
import json
video = r'''Z:\\\\content-ai.worktrees\\\\agents-content-analysis-recommendation-system\\\\videos\\\\review_keyboard.mp4'''
audio = 'temp.wav'
print('Extracting audio...')
extract_audio(video, audio)
print('Transcribing...')
res = transcribe_with_meta(audio, language='th')
with open('tmp_whisper_result.json','w',encoding='utf-8') as fh:
    json.dump(res, fh, ensure_ascii=False, indent=2)
print('WROTE tmp_whisper_result.json')
