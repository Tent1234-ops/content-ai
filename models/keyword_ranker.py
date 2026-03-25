from sentence_transformers import SentenceTransformer, util

print("Loading Embedding Model (Keyword Ranker)...")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def rank_keywords(text, keywords):
    if not keywords:
        return []

    text_embedding = model.encode(text, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    scores = util.cos_sim(text_embedding, keyword_embeddings)[0]

    ranked = sorted(
        [
            {
                "keyword": kw,
                "score": float(score)  # 🔥 fix numpy
            }
            for kw, score in zip(keywords, scores.cpu().numpy())
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked