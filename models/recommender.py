from sentence_transformers import SentenceTransformer, util

print("Loading Embedding Model (Recommender)...")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def recommend_content(user_text, dataset_texts, top_k=3):
    if not dataset_texts:
        return []

    user_emb = model.encode(user_text, convert_to_tensor=True)
    dataset_emb = model.encode(dataset_texts, convert_to_tensor=True)

    scores = util.cos_sim(user_emb, dataset_emb)[0]

    ranked = sorted(
        [
            {
                "text": text,
                "score": float(score)  # 🔥 fix ตรงนี้
            }
            for text, score in zip(dataset_texts, scores.cpu().numpy())
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:top_k]