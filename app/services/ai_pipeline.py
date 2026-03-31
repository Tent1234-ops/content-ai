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
        "เอาจริง", "แน่นอน", "แบบ", "คือ",
        "ก็", "นะ", "ครับ", "ค่ะ", "อ่า"
    ]

    for w in noise_words:
        text = text.replace(w, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 🔥 keyword cleaning (สำคัญสุด)
# =========================
def clean_keywords(keywords):
    cleaned = []

    for kw in keywords:
        kw = kw.strip()

        if len(kw) < 3:
            continue

        # ❌ generic ทิ้ง
        if kw in ["keyboard", "สินค้า", "ตัว", "ของ"]:
            continue

        # ❌ adjective ลอย
        if re.search(r"(เท่|ดี|มาก|สุด|สวย)$", kw):
            continue

        # ❌ ซ้ำคำ
        if len(set(kw.split())) == 1 and len(kw.split()) > 1:
            continue

        # ❌ ตัดคำมั่วจาก ASR
        if re.search(r"[ก-๙]{1,2}\s", kw):
            continue

        cleaned.append(kw)

    return list(dict.fromkeys(cleaned))


# =========================
# 🔥 domain enrichment (ปรับให้แม่น)
# =========================
def enrich_keywords(text, keywords):
    extra = []

    # keyboard domain (ตอนนี้ focus)
    if "keyboard" in text or "คีย์บอร์ด" in text:
        extra.append("mechanical keyboard")

    if "75" in text:
        extra.append("75% keyboard")

    if "red switch" in text:
        extra.append("red switch")

    if "linear" in text:
        extra.append("linear switch")

    if "pbt" in text or "double shot" in text:
        extra.append("pbt keycaps")

    if "rgb" in text or "ไฟ" in text:
        extra.append("rgb lighting")

    if "gasket" in text:
        extra.append("gasket mount")

    if "foam" in text:
        extra.append("foam mod")

    return list(dict.fromkeys(keywords + extra))


# =========================
# 🔥 classify (สำคัญมาก)
# =========================
def classify_keywords(keywords):
    content = []
    metadata = []
    entity = []
    other = []

    CONTENT_HINTS = [
        "switch", "swap", "rgb", "gasket",
        "latency", "sound", "typing",
        "battery", "performance",
        "weight", "size", "keyboard",
        "keycaps", "foam"
    ]

    METADATA_HINTS = [
        "windows", "mac", "ios", "android"
    ]

    for kw in keywords:

        # ENTITY (model)
        if re.search(r"[a-z]+\d{2,4}", kw):
            entity.append(kw)

        # metadata
        elif kw in METADATA_HINTS:
            metadata.append(kw)

        # content จริง
        elif any(hint in kw for hint in CONTENT_HINTS):
            content.append(kw)

        else:
            other.append(kw)

    return {
        "content": list(dict.fromkeys(content)),
        "metadata": list(dict.fromkeys(metadata)),
        "entity": list(dict.fromkeys(entity)),
        "other": list(dict.fromkeys(other))
    }


# =========================
# 🔥 MAIN PIPELINE (Production Ready)
# =========================
def analyze_video(video_path: str):

    # 1. audio
    audio_path = "temp.wav"
    extract_audio(video_path, audio_path)

    # 2. speech → text
    transcript = transcribe(audio_path)

    # 3. clean
    clean_transcript = clean_text(transcript)

    # 4. candidate (กว้าง)
    candidate_keywords = extract_keywords(clean_transcript)

    # 5. semantic filter
    semantic_filtered = semantic_keywords(
        clean_transcript,
        candidate_keywords,
        top_k=20
    )

    # 6. merge
    merged_keywords = list(dict.fromkeys(
        semantic_filtered + candidate_keywords
    ))

    # 7. clean keyword
    cleaned_keywords = clean_keywords(merged_keywords)

    # 8. enrich domain
    enriched_keywords = enrich_keywords(
        clean_transcript,
        cleaned_keywords
    )

    # 9. 🔥 remove metadata BEFORE rank
    filtered_keywords = [
        kw for kw in enriched_keywords
        if kw not in ["mac", "windows", "ios", "android"]
    ]

    # 10. ranking
    ranked_keywords = rank_keywords(
        clean_transcript,
        filtered_keywords
    )

    top_keywords = [k["keyword"] for k in ranked_keywords[:10]]

    # 11. classify
    classified = classify_keywords(enriched_keywords)

    # 12. summary
    summary = summarize_text(
        clean_transcript,
        keywords=top_keywords
    )

    # =========================
    # 🔥 FINAL OUTPUT (DB READY)
    # =========================
    result = {
        "transcript": transcript,

        "analysis": {
            "summary": summary,

            # UI
            "top_keywords": ranked_keywords[:10],

            # 🔥 ใช้ AI logic ต่อ
            "content_keywords": classified["content"],

            # ❌ ไม่ใช้แนะนำ
            "metadata": classified["metadata"],

            # model name
            "entities": classified["entity"],

            # dataset
            "all_keywords": enriched_keywords,

            # debug
            "candidates": candidate_keywords
        }
    }

    return convert_numpy(result)