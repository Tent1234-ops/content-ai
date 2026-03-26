from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

print("Loading Universal Keyword Extractor V6...")

thai_stop = set(thai_stopwords())

# =========================
# 🔥 noise words
# =========================
NOISE_WORDS = {
    "แน่นอน", "เอาจริง", "ส่วนตัว", "คือ", "แบบ",
    "มาก", "เลย", "โอเค", "ครับ", "ค่ะ", "ใช้ได้",
    "ตัว", "นี้", "นั้น"
}

# =========================
# 🔥 speech error patterns
# =========================
BAD_PATTERNS = [
    r"^รอก",
    r"เฟล",
]

# =========================
# Normalize
# =========================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^ก-๙a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# Tokenizer
# =========================
def tokenize(text):
    tokens = word_tokenize(text, engine="newmm")

    tokens = [
        t for t in tokens
        if t not in thai_stop
        and t not in NOISE_WORDS
        and len(t) > 2
        and not t.isnumeric()
    ]

    return tokens

# =========================
# TF-IDF
# =========================
def tfidf_keywords(text, corpus):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1,2),
        max_features=500
    )

    X = vectorizer.fit_transform(corpus + [text])
    feature_names = vectorizer.get_feature_names_out()
    doc_vector = X[-1].toarray()[0]

    scores = list(zip(feature_names, doc_vector))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [k for k, v in scores if v > 0]

# =========================
# 🔥 keyword เพี้ยน
# =========================
def is_bad_keyword(kw):
    # speech error
    for p in BAD_PATTERNS:
        if re.search(p, kw):
            return True

    # สั้นเกิน
    if len(kw) <= 2:
        return True

    return False

# =========================
# 🔥 Filter คุณภาพต่ำ
# =========================
def filter_keywords(keywords):
    clean = []

    for kw in keywords:
        words = kw.split()

        # ❌ ยาวเกิน
        if len(words) > 2:
            continue

        # ❌ มี noise
        if any(w in NOISE_WORDS for w in words):
            continue

        # ❌ เพี้ยน
        if is_bad_keyword(kw):
            continue

        # ❌ ซ้ำคำ
        if len(set(words)) != len(words):
            continue

        clean.append(kw)

    return clean

# =========================
# 🔥 Boost (generic ไม่ผูก domain)
# =========================
def boost_keywords(keywords, text):
    boosted = []

    for kw in keywords:
        score = 0

        # อยู่ใน text จริง → สำคัญ
        if kw in text:
            score += 2

        # phrase > word → สำคัญกว่า
        if len(kw.split()) > 1:
            score += 1

        boosted.append((kw, score))

    boosted = sorted(boosted, key=lambda x: x[1], reverse=True)

    return [k for k, _ in boosted]

# =========================
# 🔥 Deduplicate concept
# =========================
def deduplicate_keywords(keywords):
    final = []
    seen = set()

    for kw in keywords:
        base = kw.split()[-1]

        if base not in seen:
            final.append(kw)
            seen.add(base)

    return final

# =========================
# Extract model
# =========================
def extract_model(text):
    return list(set(re.findall(r"[a-z]+\d{2,4}", text)))

# =========================
# MAIN
# =========================
def extract_keywords(text, corpus=None):
    try:
        text = normalize_text(text)

        if corpus is None:
            corpus = [text]

        keywords = []

        # 1. TF-IDF
        tfidf_k = tfidf_keywords(text, corpus)

        # 2. filter
        tfidf_k = filter_keywords(tfidf_k)

        # 3. boost
        tfidf_k = boost_keywords(tfidf_k, text)

        # 4. model name (สำคัญ)
        keywords += extract_model(text)

        # 5. merge
        keywords += tfidf_k

        # 6. deduplicate concept
        keywords = deduplicate_keywords(keywords)

        # 7. remove dup
        keywords = list(dict.fromkeys(keywords))

        return keywords[:10]

    except Exception as e:
        print("Keyword error:", e)
        return []