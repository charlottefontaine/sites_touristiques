"""
Cleans the corpus (from corpus_json.json), builds the matrix
df_freq_terms.csv and calculates the TF-IDF matrix aggregated by city.
Inputs : 
- data/processed/corpus_json.json : Raw corpus in JSON, one line per page (city, url, text columns, etc.)

Outputs : 
- data/processed/df_freq_terms.csv  : Frequency matrix (documents x terms) with a 'city' column.

- data/processed/tfidf_by_city_norm.csv : Average TF-IDF per city, each row normalized to 1.

- data/text_analysis/{city}_top_words.png : Barplots of top TF-IDF terms by city.
"""
from Utils import * 
from parameters import * 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer 
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TEXT_ANALYSIS_DIR = os.path.join(BASE_DIR, "text_analysis")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TEXT_ANALYSIS_DIR, exist_ok=True)

def build_df_freq_terms(dict_tokens: dict) -> pd.DataFrame:
    """
    Constructs a cleaned term-document matrix from tokens.
    Settings
    ----------
    dict_tokens: dict
        {(city, url): [tokens]}
    Returns
    -------
    pd.DataFrame
        Frequency matrix (documents x terms) with 'city' column.
    """
    cities = {key[0] for key in dict_tokens.keys()}
    print(f"Cities present: {sorted(cities)}")
    print(f"Total entries: {len(dict_tokens)}")

    # TDM 
    df_tdm = build_term_document_matrix(dict_tokens)
    print(f"Term-document matrix shape: {df_tdm.shape}")

    # Cleaning Pipeline 
    df_tdm = clean_term_document_matrix(df_tdm, corrector=None, keep_numbers=False)
    df_tdm = normalize_social_media_terms(df_tdm)
    df_tdm = remove_nltk_stopwords(df_tdm)
    df_tdm = lemmatize_matrix_nltk(df_tdm)
    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)
    df_tdm = remove_miniwords(df_tdm, min_length=3)
    df_freq_terms = filter_terms_by_frequency(df_tdm)

    print(f"Final shape after filtering: {df_freq_terms.shape}")

    # Add city
    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    return df_freq_terms

def compute_tfidf(df_freq_terms: pd.DataFrame):
    """
   Calculates TF-IDF at the document level and average TF-IDF per city.
    Returns
    -------
    df_tfidf: pd.DataFrame -> TF-IDF by document + 'city'.
    tfidf_by_city_norm: pd.DataFrame -> Average TF-IDF per city, lines normalized to 1.
    """
    terms_only = [c for c in df_freq_terms.columns if c != "city"]
    tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(df_freq_terms[terms_only].values)

    df_tfidf = pd.DataFrame(
        X_tfidf.toarray(),
        columns=terms_only,
        index=df_freq_terms.index,
    )
    df_tfidf["city"] = df_freq_terms["city"].values

    tfidf_by_city = df_tfidf.groupby("city").mean()
    tfidf_by_city_norm = tfidf_by_city.div(tfidf_by_city.sum(axis=1), axis=0)
    return df_tfidf, tfidf_by_city_norm

def plot_top_words_per_city(tfidf_by_city_norm: pd.DataFrame,top_n: int = 10,output_folder: str = TEXT_ANALYSIS_DIR,):
    """
    Saving barplots of top TF-IDF terms by city.
    """
    os.makedirs(output_folder, exist_ok=True)
    for city in tfidf_by_city_norm.index:
        top_words = (
            tfidf_by_city_norm.loc[city]
            .sort_values(ascending=False)
            .head(top_n)
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_words.values, y=top_words.index, palette="Reds_r")
        plt.title(f"Top {top_n} words for {city}")
        plt.xlabel("Normalized TF-IDF")
        plt.ylabel("Words")
        plt.tight_layout()
        out_path = os.path.join(output_folder, f"{city}_top_words.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Plot created: {out_path}")


def main():
    # 1) global corpus -> df_freq_terms.csv
    json_path = os.path.join(PROCESSED_DIR, "corpus_json.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found. Run corpus_analysis.py first."
        )

    dict_tokens_full = tokenize_json_by_city_url(json_path)
    df_freq_terms_full = build_df_freq_terms(dict_tokens_full)

    freq_path = os.path.join(PROCESSED_DIR, "df_freq_terms.csv")
    df_freq_terms_full.to_csv(freq_path, index=False)
    print(f"Saved global frequency matrix: {freq_path}")

    # 2) TF-IDF pper city
    df_tfidf, tfidf_by_city_norm = compute_tfidf(df_freq_terms_full)
    tfidf_city_path = os.path.join(PROCESSED_DIR, "tfidf_by_city_norm.csv")
    tfidf_by_city_norm.to_csv(tfidf_city_path)
    print(f"Saved city-level TF-IDF: {tfidf_city_path}")

    # 3) Barplots TF-IDF per city 
    plot_top_words_per_city(tfidf_by_city_norm, top_n=10)

if __name__ == "__main__":
    main()