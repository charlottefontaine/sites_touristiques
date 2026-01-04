from Utils import *
from parameters import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Frequency filtering
    df_freq_terms = filter_terms_by_frequency(df_tdm)
    print(f"Final shape after filtering: {df_freq_terms.shape}")

    # Add city column
    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    os.makedirs("data/processed", exist_ok=True)
    return df_freq_terms

def get_short_terms(term_document_matrix, max_len=3):
    """
    Return list of short terms with length <= max_len
    """
    return [term for term in term_document_matrix.columns if len(term) <= max_len]

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

def plot_heatmap_top_words(df_freq_terms, zone_name, top_n=10, output_folder="data/processed/plots"):
    os.makedirs(output_folder, exist_ok=True)
    terms_only = [c for c in df_freq_terms.columns if c != "city"]
    top_words_zone = df_freq_terms[terms_only].sum().sort_values(ascending=False).head(top_n).index
    df_heatmap = df_freq_terms.groupby("city")[top_words_zone].sum()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap,annot=True,cmap="YlOrRd",fmt=".0f")
    plt.title(f"Heatmap des {top_n} mots les plus fréquents - Zone: {zone_name}")
    plt.xlabel("Mots")
    plt.ylabel("Villes")
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/heatmap_{zone_name}.png", dpi=300)
    plt.close()
    print(f"Heatmap créée : {output_folder}/heatmap_{zone_name}.png")

#---------------------------------
# Part1. Dimensionality reduction
#---------------------------------

dict_tokens = tokenize_json_by_city_url("data/processed/corpus_json.json")
df_tdm = build_term_document_matrix(dict_tokens)

# Remove mini-words here for TF-IDF only
df_tdm_miniwords_removed = remove_miniwords(df_tdm, min_length=3)
print("Preview after removing miniwords:")       
print(df_tdm_miniwords_removed.head())

df_freq_terms = filter_terms_by_frequency(df_tdm_miniwords_removed)
print("Preview of term frequencies:")    
print(df_freq_terms.head(10))
print(f"Term-document matrix shape after frequency filtering: {df_freq_terms.shape}")

df_freq_terms.to_csv('data/processed/df_freq_terms.csv', index=False)

#----------------------
# Part2. Exploration
#----------------------

word_frequencies = df_freq_terms.sum(axis=0)
top_20_words_high = word_frequencies.sort_values(ascending=False).head(20)
print("Top 20 most frequent words:\n", top_20_words_high)

top_50_words_low = word_frequencies.sort_values(ascending=True).head(50)
print("Top 50 least frequent words:\n", top_50_words_low)

short_terms = get_short_terms(df_freq_terms, max_len=3)
print(f"Short terms (length <= 3): {short_terms}")  
print(f"Number of short terms: {len(short_terms)}")

#-------------------------------
# Part3. CSV file modification
#-------------------------------

df_freq_terms = df_freq_terms.copy()
df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
df_freq_terms = df_freq_terms[cols]

df_freq_terms.to_csv('data/processed/df_freq_terms.csv', index=False)
print(f"Filtered term-document matrix saved: {df_freq_terms.shape}")

terms_only = [c for c in df_freq_terms.columns if c != "city"]
if len(terms_only) == 0:
    print("Warning: no term columns found in df_freq_terms (only 'city' present?)")
else:
    word_frequencies = df_freq_terms[terms_only].sum(axis=0)
    print("Top 20 most frequent words:\n", word_frequencies.sort_values(ascending=False).head(20))
    print("Bottom 20 least frequent words:\n", word_frequencies.sort_values(ascending=True).head(20))

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

    dict_tokens_zone = tokenize_json_by_city_url(path)
    df_freq_terms_zone = main(dict_tokens_zone)
    df_terms_only = df_freq_terms_zone.drop(columns=["city"])
    df_zone = pd.DataFrame(df_terms_only.sum()).T
    output_path = f"data/processed/tdm_{zone}.csv"
    df_zone.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    
for zone, path in corpus_files.items():
    print(f"\nProcessing zone: {zone}")
    dict_tokens_z = tokenize_json_by_city_url(path)
    df_zone_full = main(dict_tokens_z)
    _, tfidf_norm = compute_tfidf(df_zone_full)
    plot_top_words_per_city(tfidf_norm, top_n=10, output_folder=f"data/processed/plots/{zone}")
    plot_heatmap_top_words(df_zone_full, zone, top_n=15, output_folder="data/processed/plots")

    df_terms_only = df_zone_full.drop(columns=["city"])
    df_sum_zone = pd.DataFrame(df_terms_only.sum()).T
    df_sum_zone.to_csv(f"data/processed/tdm_{zone}.csv", index=False)
#----------------------
# Part4. TF-IDF
#----------------------

tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
X_tfidf = tfidf_transformer.fit_transform(df_freq_terms[terms_only].values)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=terms_only, index=df_freq_terms.index)

# Add city column
df_tfidf["city"] = df_freq_terms["city"].values

print(f"TF-IDF matrix shape: {df_tfidf.shape}")
print(df_tfidf.head())

tfidf_by_city = df_tfidf.groupby("city").mean()
tfidf_by_city_norm = tfidf_by_city.div(tfidf_by_city.sum(axis=1), axis=0)

print("TF-IDF normalized by city:")
print(tfidf_by_city_norm.head())
row_sums = tfidf_by_city_norm.sum(axis=1)
print("Sum of normalized TF-IDF per city:", row_sums)

#---------------------------------
# Part5. Term-document matrix by zone
#---------------------------------

SEA_CITIES = {"Barcelona", "Lisbon", "Copenhagen", "Ostend", "Valencia"}
NOSEA_CITIES = {"Rome", "Manchester", "Cologne", "Amsterdam", "Bruges"}
SOUTH_CITIES = {"Barcelona", "Lisbon", "Rome", "Valencia"}
NORTH_CITIES = {"Amsterdam", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"}

all_freq_terms = []

for zone, path in corpus_files.items():
    dict_tokens_z = tokenize_json_by_city_url(path)
    df_zone = main(dict_tokens_z)
    all_freq_terms.append(df_zone)

print("Finally")