import subprocess
import numpy as np

from models.speech_to_text import transcribe
from models.keyword_llm import extract_keywords
from models.summarizer import summarize_text
from models.keyword_ranker import rank_keywords
from models.keyword_gap import keyword_gap
from models.recommender import recommend_content


def convert_numpy(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    return obj


def extract_audio(video_path, audio_path="temp.wav"):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def analyze_video(video_path: str):

    # 1. audio
    audio_path = "temp.wav"
    extract_audio(video_path, audio_path)

    # 2. speech → text
    transcript = transcribe(audio_path)

    # 🔥 3. keyword จาก RAW (สำคัญสุด)
    raw_keywords = extract_keywords(transcript)

    # 🔥 4. ranking จาก RAW
    ranked_keywords = rank_keywords(transcript, raw_keywords)

    keyword_list = [k["keyword"] for k in ranked_keywords]

    # 🔥 5. summary ใช้ keyword
    summary = summarize_text(transcript, keywords=keyword_list)

    # =========================
    viral_keywords = [
        "mechanical keyboard", "hot swap", "rgb",
        "gasket", "sound test", "typing feel", "latency"
    ]

    dataset_texts = [
        "รีวิว mechanical keyboard hot swap rgb gasket",
        "sound test keyboard typing feel",
        "best budget keyboard wireless rgb"
    ]

    # 6. gap
    gap_keywords = keyword_gap(keyword_list, viral_keywords)

    # 7. recommendation
    recommendations = recommend_content(summary, dataset_texts)

    result = {
        "transcript": transcript,
        "summary": summary,
        "keywords": ranked_keywords[:10],
        "keyword_gap": gap_keywords,
        "recommendations": recommendations
    }

    return convert_numpy(result)