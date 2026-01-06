"""
Extracts keyword-in-context (KWIC) concordances for selected words and cities
from the tourism corpus, and saves them as CSV files.

The script:
- loads the preprocessed corpus from data/processed/corpus.csv,
- optionally filters documents to a subset of cities,
- computes KWIC concordance lines (left context, keyword, right context)
  for a set of strategic keywords,
- saves one CSV file per keyword with all concordance lines across cities.

Inputs
- data/processed/corpus.csv
    Expected columns: 'city', 'text'.
    'city' is used to filter and label concordance lines; 'text' is the full page text.

Outputs
- One CSV per keyword, written to:
    data/text_analysis/concordances/concordance_{keyword}.csv
  Columns: city, left, keyword, right.
- Console:
  For each keyword, prints the number of documents in the subset
  and the frequency of concordance lines per city.
"""

import os
import pandas as pd

BASE_PATH = "data/processed"
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.csv")
OUTPUT_DIR = "data/text_analysis/concordances"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_corpus(path: str, cities_filter=None) -> pd.DataFrame:
    """
    Load the corpus and optionally filter on a list of cities.

    Parameters
    ----------
    path : str
        Path to the corpus CSV file.
    cities_filter : list[str] or None
        Optional list of city names to keep.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 'city' and 'text' columns, NaN texts removed.
    """
    df = pd.read_csv(path)
    if "city" not in df.columns or "text" not in df.columns:
        raise ValueError("Expected columns 'city' and 'text' in corpus.csv")

    df = df.dropna(subset=["text"]).copy()
    df["city"] = df["city"].astype(str).str.strip()

    if cities_filter is not None:
        cities_filter = [c.strip() for c in cities_filter]
        df = df[df["city"].isin(cities_filter)].copy()

    return df


def kwic_for_token(text: str, token: str, window: int = 5):
    """
    Compute KWIC (Key Word In Context) lines for a token in a given text.

    Parameters
    ----------
    text : str
        Full text to search.
    token : str
        Target token to find.
    window : int, optional
        Number of words to the left and right to keep as context.

    Returns
    -------
    list[dict]
        List of dictionaries with keys: 'left', 'keyword', 'right'.
    """
    words = str(text).split()
    token_lower = token.lower()
    results = []

    for i, w in enumerate(words):
        if w.lower() == token_lower:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            left = " ".join(words[start:i])
            keyword = words[i]
            right = " ".join(words[i + 1 : end])
            results.append({"left": left, "keyword": keyword, "right": right})

    return results


def concordance(df_corpus: pd.DataFrame, token: str, window: int = 5) -> pd.DataFrame:
    """
    Build a concordance DataFrame for a given token over a corpus.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus with columns 'city' and 'text'.
    token : str
        Target token to find.
    window : int, optional
        Number of context words on each side.

    Returns
    -------
    pd.DataFrame
        Concordance lines with columns: city, left, keyword, right.
    """
    rows = []

    for _, row in df_corpus.iterrows():
        city = row["city"]
        text = row["text"]

        kwics = kwic_for_token(text, token, window=window)

        for k in kwics:
            rows.append(
                {
                    "city": city,
                    "left": k["left"],
                    "keyword": k["keyword"],
                    "right": k["right"],
                }
            )

    return pd.DataFrame(rows)


def main():
    # strategic words and cities to compare
    keyword_city_config = {
        "romantic": ["Bruges", "Amsterdam", "Lisbon"],
        "restaurant": ["Bruges", "Barcelona", "Manchester"],
        "card": ["Bruges", "Amsterdam", "Ostend"],
        "shopping": ["Manchester", "Bruges", "Valencia"],
    }

    window_size = 5 # to adjust

    for target_word, cities_filter in keyword_city_config.items():
        print(
            f"\n=== Concordance for '{target_word}' "
            f"(cities: {cities_filter}) ==="
        )

        df_corpus = load_corpus(CORPUS_PATH, cities_filter=cities_filter)
        print(f"Corpus subset loaded: {len(df_corpus)} documents")

        df_kwic = concordance(
            df_corpus,
            token=target_word,
            window=window_size,
        )

        if df_kwic.empty:
            print(
                f"No occurrences of '{target_word}' "
                f"found for cities {cities_filter}."
            )
            continue

        print(df_kwic["city"].value_counts())

        output_csv = os.path.join(
            OUTPUT_DIR,
            f"concordance_{target_word}.csv",
        )
        df_kwic.to_csv(output_csv, index=False)
        print(f"Concordance saved to: {output_csv}")


if __name__ == "__main__":
    main()
