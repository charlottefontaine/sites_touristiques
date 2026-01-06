"""
Part 1: VADER
    - Calculates sentiment scores (neg, neu, pos, compound) per page.
    - Save an enriched corpus with sentiment score

Part 2: Extended emotional lexicon
    - Constructs a profile of emotions per city (joy, fear, etc.).
    - Save a table aggregated by city.

Inputs
------
data/processed/corpus.csv

Outputs
-------
data/processed/corpus_with_sentiment.csv
    Corpus enriched with VADER scores.

data/text_analysis/sentiment/sentiment_profiles_by_city_extended.csv
    Emotion profiles by city (rate per 1000 words, normalized shares,
    dominant emotion).
"""
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
SENTIMENT_DIR = os.path.join(BASE_DIR, "text_analysis", "sentiment")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)

## PART 1 : VADER
# 1. Load VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# 2. Load the corpus
df = pd.read_csv("data/processed/corpus.csv") # VADER is train on a complete natural language

# 3. Initialize the sentiment thesaurus (VADER lexicon)
sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text: str):
    """
    Return VADER sentiment scores for a text.
    """
    if not isinstance(text, str):
        text = str(text)
    return sia.polarity_scores(text)

# 4. Apply to the corpus
sentiment_dicts = df["text"].apply(get_sentiment_scores)

# Convert in columns
sentiment_df = pd.DataFrame(list(sentiment_dicts))
# columns: ['neg', 'neu', 'pos', 'compound']

# 5. Join to the corpus
df_sentiment = pd.concat([df, sentiment_df], axis=1)

# 6. Save
df_sentiment.to_csv("data/processed/corpus_with_sentiment.csv", index=False)
print("Sentiment scores saved to data/processed/corpus_with_sentiment.csv")
print(df_sentiment.head())

df = pd.read_csv("data/processed/corpus_with_sentiment.csv")

sentiment_by_city = (
    df.groupby("city")[["neg", "neu", "pos", "compound"]]
      .mean()
      .reset_index())

print(sentiment_by_city)

##PART 2 : emotional profile per city 
import os
import re
import numpy as np

BASE_PATH = "data/text_analysis"
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.csv")
OUTPUT_DIR = "data/sentiment_lexicon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# emotional lexicon
SENTIMENT_LEXICON = {
    "joy": [
        "fun", "enjoy", "enjoyable", "enjoying", "enjoyment",
        "happy", "happiness", "smile", "smiles", "joy", "joyful",
        "delight", "delightful", "cheerful", "festive", "lively",
        "vibrant", "buzzing", "playful", "pleasant", "pleasure",
        "charming", "magical", "warm", "welcoming", "friendly",
        "funfilled", "fun-filled", "colourful", "colorful"
    ],
    "relax": [
        "relax", "relaxing", "relaxed", "relaxation",
        "calm", "peaceful", "peacefully", "quiet", "quietly",
        "slow", "slowly", "escape", "escapade", "getaway",
        "tranquil", "tranquility", "serene", "serenity",
        "wellness", "spa", "soothing", "unwind", "recharge",
        "rejuvenate", "rest", "restful"
    ],
    "energy": [
        "dynamic", "exciting", "excited", "excite",
        "thrilling", "thrill", "intense", "lively",
        "nightlife", "night-life", "party", "parties",
        "festival", "festivals", "event", "events",
        "buzzing", "crowded", "bustling", "pulsating",
        "vibrant", "animated", "energetic", "electric"
    ],
    "luxury": [
        "luxury", "luxurious", "exclusive", "premium",
        "elegant", "elegance", "sophisticated", "refined",
        "boutique", "highend", "high-end", "upscale",
        "five-star", "5star", "5-star", "chic", "stylish",
        "prestigious", "gourmet", "fine dining", "fine-dining"
    ],
    "family": [
        "family", "families", "kids", "children",
        "childfriendly", "child-friendly", "familyfriendly",
        "family-friendly", "together", "parents",
        "playground", "playgrounds", "play area", "play areas",
        "funforall", "fun-for-all", "all ages", "young and old"
    ],
    "adventure": [
        "adventure", "adventures", "adventurous",
        "explore", "exploring", "exploration", "discovery",
        "discover", "discovering",
        "hike", "hiking", "trail", "trails",
        "bike", "biking", "cycling", "cycle",
        "kayak", "kayaking", "canoe", "canoeing",
        "climb", "climbing", "surf", "surfing",
        "sail", "sailing", "outdoor", "outdoors"
    ],
    "culture": [
        "culture", "cultural", "heritage",
        "museum", "museums", "gallery", "galleries",
        "exhibition", "exhibitions", "art", "arts",
        "artists", "architecture", "architectural",
        "monument", "monuments", "historic", "historical",
        "history", "unesco", "world heritage",
        "theatre", "theater", "opera", "concert", "concerts"
    ],
}

# handle multi-words separately
MULTIWORD_TERMS = {}
SINGLEWORD_LEXICON = {}

for cat, words in SENTIMENT_LEXICON.items():
    singles = []
    multis = []
    for w in words:
        if " " in w or "-" in w:
            multis.append(w)
        else:
            singles.append(w)
    SINGLEWORD_LEXICON[cat] = singles
    MULTIWORD_TERMS[cat] = multis

CATEGORY_PATTERNS = {
    cat: re.compile(r"\b(" + "|".join(singles) + r")\b", flags=re.IGNORECASE)
    for cat, singles in SINGLEWORD_LEXICON.items() if singles
}

def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "city" not in df.columns or "text" not in df.columns:
        raise ValueError("Expected columns 'city' and 'text'")
    return df.dropna(subset=["text"]).copy()


def word_count(text: str) -> int:
    return len(str(text).split())


def count_category_terms(text: str) -> dict:
    """
    Count the occurrences of each category (single words + multi-words).
    Work on raw text to preserve natural expressions.
    """
    s = str(text)

    counts = {}
    for cat in SENTIMENT_LEXICON.keys():
        c_single = 0
        pattern = CATEGORY_PATTERNS.get(cat)
        if pattern is not None:
            c_single = len(pattern.findall(s))
        c_multi = 0
        for phrase in MULTIWORD_TERMS[cat]:
            c_multi += len(re.findall(re.escape(phrase), s, flags=re.IGNORECASE))
        counts[cat] = c_single + c_multi
    return counts

def compute_sentiment_profiles(df_corpus: pd.DataFrame) -> pd.DataFrame:
    df = df_corpus.copy()
    df["total_words"] = df["text"].apply(word_count)

    for cat in SENTIMENT_LEXICON.keys():
        df[cat + "_count"] = 0

    for idx, row in df.iterrows():
        counts = count_category_terms(row["text"])
        for cat, c in counts.items():
            df.at[idx, cat + "_count"] = c

    df["total_words"] = df["total_words"].replace(0, np.nan)

    for cat in SENTIMENT_LEXICON.keys():
        df[cat + "_per_1000"] = 1000 * df[cat + "_count"] / df["total_words"]

    group = df.groupby("city")
    agg_cols = {}
    for cat in SENTIMENT_LEXICON.keys():
        agg_cols["mean_" + cat + "_per_1000"] = (cat + "_per_1000", "mean")

    df_city = group.agg(**agg_cols)

    profile_cols = [c for c in df_city.columns if c.startswith("mean_")]
    profile = df_city[profile_cols].fillna(0)
    row_sums = profile.sum(axis=1).replace(0, np.nan)
    df_profile = profile.div(row_sums, axis=0)

    df_profile = df_profile.rename(
        columns={c: c.replace("mean_", "share_") for c in profile_cols}
    )

    dominant_cat = df_profile.idxmax(axis=1)
    dominant_cat = dominant_cat.str.replace("share_", "", regex=False)

    df_out = pd.concat([df_city, df_profile], axis=1)
    df_out["dominant_emotion"] = dominant_cat

    return df_out

def main():
    df_corpus = load_corpus(CORPUS_PATH)
    print(f"Corpus loaded: {len(df_corpus)} documents")

    df_city_sentiment = compute_sentiment_profiles(df_corpus)

    print("\nExtended emotional lexical profiles by city:")
    print(df_city_sentiment.round(3))

    out_csv = os.path.join(OUTPUT_DIR, "sentiment_profiles_by_city_extended.csv")
    df_city_sentiment.to_csv(out_csv)
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    main()
