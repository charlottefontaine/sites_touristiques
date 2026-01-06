"""
Auxiliary script to validate LDA themes by computing topic scores per city.

The script:
- loads the TF-IDF matrix aggregated by city (df_tfidf_by_city.csv),
- defines 6 hand-crafted topic word lists matching the LDA topics,
- computes, for each city, a raw score per topic (sum of TF-IDF over topic words),
- row-normalizes scores so that each city's topic scores sum to 1,
- identifies the dominant topic per city,
- exports the normalized topic scores and dominant topic to Excel and CSV.

Inputs
------
data/processed/df_tfidf_by_city.csv
    Rows: cities, columns: terms, values: TF-IDF scores.

Outputs
-------
data/processed/city_topic_scores.xlsx
data/processed/city_topic_scores.csv

Status
------
Complementary / validation analysis used to check that the LDA themes
match intuitive topic-word groups. 
"""
import pandas as pd
import numpy as np

PATH = "data/processed/df_tfidf_by_city.csv"
df_tfidf = pd.read_csv(PATH, index_col=0)

print("TF-IDF by city loaded")
print(df_tfidf.head())

topics = {
    "Topic 0": [
        "brugge", "bridge", "square", "street", "building",
        "hotel", "room", "water", "romantic", "sight",
        "museum", "indoor", "outdoor", "free", "read"
    ],

    "Topic 1": [
        "book", "tour", "airport", "bus", "station",
        "transport", "review", "activity", "show",
        "museum", "free", "roman", "basilica",
        "fiumicino", "piazza"
    ],

    "Topic 2": [
        "discover", "museum", "ticket", "guide", "art",
        "event", "best", "new", "visit", "winter",
        "december", "dec", "read", "find", "card"
    ],

    "Topic 3": [
        "restaurant", "food", "wine", "bar", "lunch",
        "terrace", "experience", "culture", "enjoy",
        "visit", "blog", "time", "information",
        "accessible", "ticket"
    ],

    "Topic 4": [
        "card", "tourist", "visit", "search", "main",
        "accessible", "menu", "icon", "favourite",
        "facebook", "linkedin", "http",
        "ostend", "oostende", "valència"
    ],

    "Topic 5": [
        "restaurant", "guide", "shopping", "theatre",
        "palace", "opera", "experience", "hotel",
        "family", "rating", "follow", "skip",
        "greater", "house", "lisboa"
    ]
}

topic_scores = pd.DataFrame(index=df_tfidf.index)

for topic, words in topics.items():
    valid_words = [w for w in words if w in df_tfidf.columns]

    if len(valid_words) == 0:
        print(f"No words found for topic: {topic}")
        topic_scores[topic] = 0
    else:
        topic_scores[topic] = df_tfidf[valid_words].sum(axis=1)

print("\nRaw topic scores per city:")
print(topic_scores.head())

topic_scores_norm = topic_scores.div(
    topic_scores.sum(axis=1),
    axis=0
).fillna(0)

print("\nNormalized topic scores:")
print(topic_scores_norm.head())

row_sums = topic_scores_norm.sum(axis=1)
print("\nSum of topic scores per city:")
print(row_sums)

if np.allclose(row_sums.values, 1.0):
    print("Verification OK: all rows sum to 1")
else:
    print("Warning: some rows do not sum to 1")


topic_scores_norm["dominant_topic"] = topic_scores_norm.idxmax(axis=1)

print("\nDominant topic per city:")
print(topic_scores_norm["dominant_topic"])

OUTPUT_PATH_XLSX = "data/text_analysis/classification/city_topic_scores.xlsx"
topic_scores_norm.to_excel(OUTPUT_PATH_XLSX, index=True)
print(f"\nSaved Excel file with topic scores: {OUTPUT_PATH_XLSX}")

OUTPUT_PATH = "data/text_analysis/classification/city_topic_scores.csv"
topic_scores_norm.to_csv(OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
