from openai import OpenAI

client = OpenAI()

def extract_keywords(text):

    prompt = f"""
จาก transcript นี้
ดึง keyword ของวิดีโอ

{text}

ตอบเป็น list keyword
ไม่เกิน 10 คำ
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )

    keywords = res.choices[0].message.content

    return keywords