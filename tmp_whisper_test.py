from utils.audio import extract_audio
from models.speech_to_text import transcribe_with_meta
import sys
video = r'''Z:\\\\content-ai.worktrees\\\\agents-content-analysis-recommendation-system\\\\videos\\\\review_keyboard.mp4'''
audio = 'temp.wav'
print('Extracting audio from', video)
extract_audio(video, audio)
print('Transcribing audio (this may take a while).')
res = transcribe_with_meta(audio, language='th')
print('TRANSCRIBE_RESULT:', res)
