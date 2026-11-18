# Run on Google Colab with a GPU runtime (Runtime -> Change runtime type -> T4 GPU).
# 1. Upload posts.csv to the Colab file panel.
# 2. Run this cell. It writes posts_with_sentiment.csv; download that file and send it back.
#
# Keeps the post id and all three class scores so the labels can be merged with the other labellers.

import pandas as pd
import torch
from transformers import pipeline

df = pd.read_csv("posts.csv", dtype={"id": str})
texts = df["text"].astype(str).tolist()

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    top_k=None,          # return scores for all three labels, not just the winner
)

results = sentiment_pipe(texts, batch_size=64)

def as_dict(r):
    return {x["label"].lower(): x["score"] for x in r}

scores = [as_dict(r) for r in results]
df["negative"] = [round(s.get("negative", 0.0), 4) for s in scores]
df["neutral"] = [round(s.get("neutral", 0.0), 4) for s in scores]
df["positive"] = [round(s.get("positive", 0.0), 4) for s in scores]
df["sentiment"] = df[["negative", "neutral", "positive"]].idxmax(axis=1)
df["confidence"] = df[["negative", "neutral", "positive"]].max(axis=1)

df[["id", "sentiment", "confidence", "negative", "neutral", "positive"]].to_csv(
    "posts_with_sentiment.csv", index=False)
print(df["sentiment"].value_counts())
