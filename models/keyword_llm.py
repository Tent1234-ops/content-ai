from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

print("Loading Universal Keyword Extractor V3...")

thai_stop = set(thai_stopwords())

# =========================
# 🔥 Universal Normalize
# =========================
def normalize_text(text):
    text = text.lower()

    # remove url / symbol
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^ก-๙a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 🔥 general synonym mapping (ไม่ยึด domain)
    replacements = {
        "รีวิว": "review",
        "แนะนำ": "recommend",
        "เปรียบเทียบ": "compare",
        "สอน": "tutorial",
        "วิธีทำ": "how to",
        "ดีที่สุด": "best",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()


# =========================
# 🔥 Tokenizer
# =========================
def tokenize(text):
    tokens = word_tokenize(text, engine="newmm")

    tokens = [
        t for t in tokens
        if t not in thai_stop
        and len(t) > 2
        and not t.isnumeric()
    ]

    return tokens


# =========================
# 🔥 Candidate N-grams
# =========================
def generate_ngrams(tokens, n=3):
    ngrams = []
    for i in range(len(tokens)):
        for j in range(1, n+1):
            if i + j <= len(tokens):
                phrase = " ".join(tokens[i:i+j])
                ngrams.append(phrase)
    return ngrams


# =========================
# 🔥 TF-IDF Core Engine
# =========================
def tfidf_keywords(text, corpus):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1,3),   # 🔥 สำคัญมาก (phrase)
        max_features=1000,
        min_df=1
    )

    X = vectorizer.fit_transform(corpus + [text])
    feature_names = vectorizer.get_feature_names_out()

    doc_vector = X[-1].toarray()[0]

    scores = list(zip(feature_names, doc_vector))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [k for k, v in scores[:30] if v > 0]


# =========================
# 🔥 Dynamic Domain Detection
# =========================
def detect_domain(text):
    domain_keywords = {
        "tech": ["review", "spec", "fps", "keyboard", "gpu", "มือถือ"],
        "beauty": ["skin", "cream", "หน้า", "ผิว", "makeup"],
        "food": ["กิน", "อร่อย", "menu", "food", "restaurant"],
        "education": ["tutorial", "learn", "สอน", "how to"],
        "entertainment": ["หนัง", "เพลง", "reaction", "funny"]
    }

    scores = {}

    for domain, keywords in domain_keywords.items():
        score = sum(1 for k in keywords if k in text)
        scores[domain] = score

    # fallback ถ้าไม่เข้า domain ไหนเลย
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "general"

    return best_domain


# =========================
# 🔥 Dynamic Domain Boost
# =========================
def boost_keywords(keywords, text, domain):
    boosted = []

    # 🔥 pattern-based boost (ไม่ fix domain)
    patterns = [
        r"\b\w+\d{2,4}\b",     # model เช่น a520, ip14
        r"\bbest\b",
        r"\breview\b",
        r"\bvs\b",
        r"\bhow to\b"
    ]

    important_phrases = []

    for p in patterns:
        matches = re.findall(p, text)
        important_phrases.extend(matches)

    for kw in keywords:
        if kw in important_phrases:
            boosted.insert(0, kw)
        else:
            boosted.append(kw)

    return boosted


# =========================
# 🔥 Extract model (generic)
# =========================
def extract_model(text):
    return list(set(re.findall(r"[a-z]+\d{2,4}", text)))


# =========================
# 🔥 Main Pipeline
# =========================
def extract_keywords(text, corpus=None):
    try:
        text = normalize_text(text)

        if corpus is None:
            corpus = [text]

        keywords = []

        # 1. detect domain
        domain = detect_domain(text)

        # 2. TF-IDF (core)
        tfidf_k = tfidf_keywords(text, corpus)

        # 3. model extraction (generic)
        keywords += extract_model(text)

        # 4. domain boost (dynamic)
        tfidf_k = boost_keywords(tfidf_k, text, domain)

        keywords += tfidf_k

        # 5. clean
        keywords = list(dict.fromkeys(keywords))

        return keywords[:10]

    except Exception as e:
        print("Keyword error:", e)
        return []