from transformers import pipeline

print("Loading summarizer...")

summarizer = pipeline(
    "text-generation",
    model="google/mt5-small"
)


def summarize(text):

    prompt = "summarize this text in thai: " + text

    result = summarizer(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )

    return result[0]["generated_text"]