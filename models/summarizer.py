from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

print("Loading Structured Summarizer V8...")

model_name = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# =========================
# 🔥 Clean (ไม่ทำลาย context)
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 🔥 Smart chunk
# =========================
def split_chunks(text, max_len=280):
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(" ".join(current)) > max_len:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# =========================
# 🔥 NEW: Prompt รองรับ keyword context
# =========================
def build_prompt(chunk, keywords=None):
    kw = ", ".join(keywords[:10]) if keywords else ""

    return f"""
สรุปข้อความต่อไปนี้ให้สั้นที่สุด

เงื่อนไข:
- ต้องมี keyword ถ้าเกี่ยวข้อง: {kw}
- เน้น: ชื่อสินค้า รุ่น ฟีเจอร์
- ห้ามเขียนยาว
- ตอบเป็น 1-2 ประโยคเท่านั้น

ข้อความ:
{chunk}

คำตอบ:
"""


# =========================
# 🔥 Clean output
# =========================
def clean_output(text):
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# 🔥 Main summarize
# =========================
def summarize_text(text, keywords=None):
    try:
        text = clean_text(text)
        chunks = split_chunks(text)

        summaries = []

        # 🔥 กัน keyword พัง
        if keywords:
            keywords = [str(k) for k in keywords]

        for chunk in chunks:
            prompt = build_prompt(chunk, keywords)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=4,                 # ✅ ใช้ beam → stable
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = clean_output(result)

            if result:
                summaries.append(result)

        # =========================
        # 🔥 fallback กัน model fail
        # =========================
        if not summaries:
            if keywords:
                return " ".join(keywords[:10])
            return text[:300]

        # =========================
        # 🔥 merge + final refine
        # =========================
        merged = " ".join(summaries)

        final_prompt = f"""
สรุปข้อความนี้ให้เหลือประโยคเดียว:

{merged}

คำตอบ:
"""

        inputs = tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=4,
            repetition_penalty=1.2
        )

        final_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_summary = clean_output(final_summary)

        return final_summary

    except Exception as e:
        print("Summary error:", e)
        return text[:200]