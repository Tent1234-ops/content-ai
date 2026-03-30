import subprocess
import numpy as np
import re

from models.speech_to_text import transcribe
from models.keyword_llm import extract_keywords
from models.semantic_keyword import semantic_keywords
from models.summarizer import summarize_text
from models.keyword_ranker import rank_keywords


# =========================
# utils
# =========================
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


# =========================
# clean text
# =========================
def clean_text(text: str):
    text = text.lower()

    noise_words = [
        "เอาจริง", "แน่นอน", "แบบ", "คือ", "ก็", "นะ", "ครับ", "ค่ะ"
    ]

    for w in noise_words:
        text = text.replace(w, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 🔥 post-filter keyword (สำคัญมาก)
# =========================
def clean_keywords(keywords):
    cleaned = []

    for kw in keywords:
        kw = kw.strip()

        # ❌ ตัด keyword ขยะ
        if len(kw) < 3:
            continue

        # ❌ pattern แปลก เช่น "คีย์บอร์ด เท่"
        if re.search(r"(เท่|ดี|มาก|สุด)$", kw):
            continue

        # ❌ keyword ซ้ำคำ
        if len(set(kw.split())) == 1 and len(kw.split()) > 1:
            continue

        cleaned.append(kw)

    return list(dict.fromkeys(cleaned))


# =========================
# MAIN PIPELINE
# =========================
def analyze_video(video_path: str):

    # =========================
    # 1. audio
    # =========================
    audio_path = "temp.wav"
    extract_audio(video_path, audio_path)

    # =========================
    # 2. speech → text
    # =========================
    transcript = transcribe(audio_path)

    # =========================
    # 3. clean
    # =========================
    clean_transcript = clean_text(transcript)

    # =========================
    # 4. candidate (กว้าง)
    # =========================
    candidate_keywords = extract_keywords(clean_transcript)

    # =========================
    # 5. semantic filter (คัดจริง)
    # =========================
    semantic_filtered = semantic_keywords(
        clean_transcript,
        candidate_keywords,
        top_k=20
    )

    # =========================
    # 6. merge + clean
    # =========================
    merged_keywords = list(dict.fromkeys(
        semantic_filtered + candidate_keywords
    ))

    final_keywords = clean_keywords(merged_keywords)

    # =========================
    # 7. ranking (embedding)
    # =========================
    ranked_keywords = rank_keywords(
        clean_transcript,
        final_keywords
    )

    # 🔥 เอา top จริง
    top_keywords = [k["keyword"] for k in ranked_keywords[:10]]

    # =========================
    # 8. summary (ใช้ top keyword เท่านั้น)
    # =========================
    summary = summarize_text(
        clean_transcript,
        keywords=top_keywords
    )

    # =========================
    # 9. STRUCTURED OUTPUT (🔥 ใช้เข้า DB)
    # =========================
    result = {
        "transcript": transcript,

        "analysis": {
            "summary": summary,

            # 🔥 keyword ที่สำคัญจริง
            "top_keywords": ranked_keywords[:10],

            # 🔥 keyword ทั้งหมด (dataset ใช้ตัวนี้)
            "all_keywords": final_keywords,

            # 🔥 raw candidate (debug/ปรับ model)
            "candidates": candidate_keywords
        }
    }

    return convert_numpy(result)