from Utils import *
from parameters import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

def get_short_terms(term_document_matrix, max_len=3):
    """
    Return list of short terms with length <= max_len
    """
    return [term for term in term_document_matrix.columns if len(term) <= max_len]

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

SEA_CITIES = {"Barcelona", "Lisbon", "Copenhagen", "Ostend", "Valencia"}
NOSEA_CITIES = {"Rome", "Manchester", "Cologne", "Amsterdam", "Bruges"}
SOUTH_CITIES = {"Barcelona", "Lisbon", "Rome", "Valencia"}
NORTH_CITIES = {"Amsterdam", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"}

# Main processing per zone
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


cities = df_all_cities["city"].unique()

for city in cities:
    df_city = df_all_cities[df_all_cities["city"] == city]
    terms_only = [c for c in df_city.columns if c != "city"]
    word_counts = df_city[terms_only].sum(axis=0)
    top10 = word_counts.sort_values(ascending=False).head(10)
    print(f"\nTop 10 most frequent words in {city}:")
    print(top10)

print("Finally")
