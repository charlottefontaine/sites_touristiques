from Utils import * 
from parameters import * 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer 
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Part 1 . Dimensionality reduction
# ---------------------------------

def main(dict_tokens):
  
    cities = set(key[0] for key in dict_tokens.keys())
    print(f"Cities present: {sorted(cities)}")
    print(f"Total entries: {len(dict_tokens)}")

    df_tdm = build_term_document_matrix(dict_tokens)
    print(f"Term-document matrix shape: {df_tdm.shape}")
    print(df_tdm.head())

    df_tdm = clean_term_document_matrix(df_tdm, None, keep_numbers=False)
    df_tdm = normalize_social_media_terms(df_tdm)
    df_tdm = remove_nltk_stopwords(df_tdm)
    df_tdm = lemmatize_matrix_nltk(df_tdm)
    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)  
    df_tdm = remove_miniwords(df_tdm, min_length=3)
    df_freq_terms = filter_terms_by_frequency(df_tdm)
    print(f"Final shape after filtering: {df_freq_terms.shape}")

# ---------------------------------
# Part 2 . CSV preparation
# ---------------------------------

    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    os.makedirs("data/processed", exist_ok=True)
    return df_freq_terms

def compute_tfidf(df_freq_terms):
    terms_only = [c for c in df_freq_terms.columns if c != "city"]
    tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(df_freq_terms[terms_only].values)

    df_tfidf = pd.DataFrame(
        X_tfidf.toarray(),
        columns=terms_only,
        index=df_freq_terms.index
    )
    df_tfidf["city"] = df_freq_terms["city"].values

    tfidf_by_city = df_tfidf.groupby("city").mean()
    tfidf_by_city_norm = tfidf_by_city.div(tfidf_by_city.sum(axis=1), axis=0)
    return df_tfidf, tfidf_by_city_norm

def plot_top_words_per_city(tfidf_by_city_norm, top_n=10, output_folder="data/processed/plots"):
    os.makedirs(output_folder, exist_ok=True)
    for city in tfidf_by_city_norm.index:
        top_words = tfidf_by_city_norm.loc[city].sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_words.values, y=top_words.index, palette="Reds_r")
        plt.title(f"Top {top_n} mots pour {city}")
        plt.xlabel("TF-IDF normalisé")
        plt.ylabel("Mots")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{city}_top_words.png", dpi=300)
        plt.close()
        print(f"Plot créé : {output_folder}/{city}_top_words.png")

# -----------------------------
# Main processing for 4 corpus
# -----------------------------
corpus_files = {
    "sea": "data/processed/corpus_sea.json",
    "nosea": "data/processed/corpus_nosea.json",
    "north": "data/processed/corpus_north.json",
    "south": "data/processed/corpus_south.json"
}

for zone, path in corpus_files.items():
    print(f"\nProcessing zone: {zone}")

    dict_tokens = tokenize_json_by_city_url(path)
    df_freq_terms = main(dict_tokens)
    df_terms_only = df_freq_terms.drop(columns=["city"])
    df_zone = pd.DataFrame(df_terms_only.sum()).T
    output_path = f"data/processed/tdm_{zone}.csv"
    df_zone.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    #  build df_freq_terms.csv for LDA analysis
    dict_tokens_full = tokenize_json_by_city_url("data/processed/corpus_json.json")
    df_freq_terms_full = main(dict_tokens_full)
    df_freq_terms_full.to_csv("data/processed/df_freq_terms.csv", index=False)
    print("Saved global frequency matrix: data/processed/df_freq_terms.csv ")