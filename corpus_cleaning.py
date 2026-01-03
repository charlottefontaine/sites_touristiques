from Utils import *
from parameters import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import os

# ---------------------------------
# Part 1 . Term-document matrix
# ---------------------------------
def main(dict_tokens):
    cities = set(key[0] for key in dict_tokens.keys())
    print(f"Cities present: {sorted(cities)}")
    print(f"Total entries: {len(dict_tokens)}")

    df_tdm = build_term_document_matrix(dict_tokens)
    print(f"Term-document matrix shape: {df_tdm.shape}")

    df_tdm = clean_term_document_matrix(df_tdm, None, keep_numbers=False)
    df_tdm = normalize_social_media_terms(df_tdm)
    df_tdm = remove_nltk_stopwords(df_tdm)
    df_tdm = lemmatize_matrix_nltk(df_tdm)
    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)
    df_tdm = remove_miniwords(df_tdm, min_length=3)
    df_freq_terms = filter_terms_by_frequency(df_tdm)
    print(f"Final shape after filtering: {df_freq_terms.shape}")

    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    os.makedirs("data/processed", exist_ok=True)
    return df_freq_terms


SEA_CITIES = {"Barcelona", "Lisbon", "Copenhagen", "Ostend", "Valencia"}
NOSEA_CITIES = {"Rome", "Manchester", "Cologne", "Amsterdam", "Bruges"}
SOUTH_CITIES = {"Barcelona", "Lisbon", "Rome", "Valencia"}
NORTH_CITIES = {"Amsterdam", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"}


# ---------------------------------
# Main processing
# ---------------------------------
corpus_files = {
    "sea": "data/processed/corpus_sea.json",
    "nosea": "data/processed/corpus_nosea.json",
    "north": "data/processed/corpus_north.json",
    "south": "data/processed/corpus_south.json"
}

all_freq_terms = []

for zone, path in corpus_files.items():
    dict_tokens = tokenize_json_by_city_url(path)
    df_zone = main(dict_tokens)
    all_freq_terms.append(df_zone)

df_all_cities = pd.concat(all_freq_terms, ignore_index=True)
df_all_cities.to_csv("data/processed/df_freq_terms_all.csv", index=False)

print("Finally")
