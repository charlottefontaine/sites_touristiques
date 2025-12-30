import os
import re
from collections import Counter, defaultdict

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import bigrams

from Utils import tokenize_json_by_city_url
from parameters import project_stopwords


BASE_PATH = "data/processed"
JSON_PATH = os.path.join(BASE_PATH, "corpus_json.json")
OUTPUT_PREFIX = BASE_PATH  # fichiers CSV dans data/processed


# Stopwords
try:
    EN_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    EN_STOPWORDS = set(stopwords.words("english"))

PROJECT_STOPWORDS = {sw.lower().strip() for sw in project_stopwords}


def clean_token_keep_order(token: str):
    t = token.lower()
    t = re.sub(r"[^a-z]", "", t)
    if not t:
        return None
    if t in EN_STOPWORDS:
        return None
    if t in PROJECT_STOPWORDS:
        return None
    if len(t) < 2:
        return None
    return t


def build_clean_tokens_by_doc(json_path: str):
    """
    Retourn a dict (city, url) -> list of cleaned tokens.
    """
    raw_tokens = tokenize_json_by_city_url(json_path)
    clean_dict = {}

    for key, tokens in raw_tokens.items():
        cleaned = []
        for tok in tokens:
            ct = clean_token_keep_order(tok)
            if ct is not None:
                cleaned.append(ct)
        if cleaned:
            clean_dict[key] = cleaned

    return clean_dict


def get_bigrams_by_city(tokens_by_doc: dict) -> dict:
    """
    compute bigrams per city 
    """
    by_city = defaultdict(Counter)

    for (city, url), tokens in tokens_by_doc.items():
        by_city[city].update(list(bigrams(tokens)))

    return by_city


def filter_strategic_bigrams_by_city(bigrams_city: dict,
                                     target_words: list[str],
                                     min_freq: int = 3) -> dict:
    """
    keep strategic bigrams
    """
    target_set = {w.lower() for w in target_words}
    result = {}

    for city, counter in bigrams_city.items():
        filtered = {}
        for (w1, w2), f in counter.items():
            if f < min_freq:
                continue
            if (w1 in target_set) or (w2 in target_set):
                filtered[(w1, w2)] = f

        result[city] = Counter(filtered)

    return result


def save_bigrams_by_city(bigrams_city: dict,
                         out_prefix: str,
                         suffix: str = "strategic"):
    """
    save 
    """
    for city, counter in bigrams_city.items():
        if not counter:
            continue

        df_city = (
            pd.DataFrame(
                [(w1, w2, f) for (w1, w2), f in counter.items()],
                columns=["word1", "word2", "frequency"],
            )
            .sort_values("frequency", ascending=False)
        )

        safe_city = re.sub(r"[^A-Za-z0-9]+", "_", city)
        path = f"{out_prefix}/bigrams_{safe_city}_{suffix}.csv"
        df_city.to_csv(path, index=False)

        print(f"Top strategic bigrams for {city}:")
        print(df_city.head(10))


def main():
    tokens_by_doc = build_clean_tokens_by_doc(JSON_PATH)
    print(f"Number of documents with cleaned tokens: {len(tokens_by_doc)}")

    # 1) Bigrams per city 
    bi_city = get_bigrams_by_city(tokens_by_doc)

    # 2) strategic bigrams
    strategic_words = ["romantic", "restaurant", "card", "shopping"]
    bi_city_strategic = filter_strategic_bigrams_by_city(
        bi_city,
        target_words=strategic_words,
        min_freq=3,
    )

    # 3) Save
    save_bigrams_by_city(
        bi_city_strategic,
        out_prefix=OUTPUT_PREFIX,
        suffix="strategic",
    )


if __name__ == "__main__":
    main()
