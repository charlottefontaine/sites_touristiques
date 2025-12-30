import os
import pandas as pd

BASE_PATH = "data/processed"
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.csv")
OUTPUT_DIR = "data/concordances"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_corpus(path: str, cities_filter=None) -> pd.DataFrame:
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
    words = str(text).split()
    token_lower = token.lower()
    results = []

    for i, w in enumerate(words):
        if w.lower() == token_lower:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            left = " ".join(words[start:i])
            keyword = words[i]
            right = " ".join(words[i + 1:end])
            results.append({"left": left, "keyword": keyword, "right": right})
    return results


def concordance(df_corpus: pd.DataFrame,
                token: str,
                window: int = 5) -> pd.DataFrame:
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
    # Mots stratégiques + villes à comparer
    keyword_city_config = {
        "romantic": ["Bruges", "Amsterdam", "Lisbon"],
        "restaurant": ["Bruges", "Barcelona", "Manchester"],
        "card": ["Bruges", "Amsterdam", "Ostend"],
        "shopping": ["Manchester", "Bruges", "Valencia"],
    }

    window_size = 5
    max_per_city_for_sample = 20  # à ajuster (10, 20, 30...)

    for target_word, cities_filter in keyword_city_config.items():
        print(f"\n=== Concordance for '{target_word}' "
              f"(cities: {cities_filter}) ===")

        df_corpus = load_corpus(CORPUS_PATH, cities_filter=cities_filter)
        print(f"Corpus subset loaded: {len(df_corpus)} documents")

        df_kwic = concordance(
            df_corpus,
            token=target_word,
            window=window_size,
        )

        if df_kwic.empty:
            print(f"No occurrences of '{target_word}' "
                  f"found for cities {cities_filter}.")
            continue

        # Stats brutes
        print(df_kwic["city"].value_counts())

        # CSV complet
        full_csv = os.path.join(
            OUTPUT_DIR,
            f"concordance_{target_word}.csv",
        )
        df_kwic.to_csv(full_csv, index=False)
        print(f"Full concordance saved to: {full_csv}")

        # Échantillon équilibré : max N lignes par ville
        df_sample = (
            df_kwic
            .groupby("city", group_keys=False)
            .head(max_per_city_for_sample)
        )

        sample_csv = os.path.join(
            OUTPUT_DIR,
            f"concordance_{target_word}_sample.csv",
        )
        df_sample.to_csv(sample_csv, index=False)
        print(f"Sample concordance saved to: {sample_csv}")


if __name__ == "__main__":
    main()

