from Utils import *
from parameters import * 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer 

def main(dict_tokens):

    # ---------------------------------
    # Part 1 . Dimensionality reduction
    # ---------------------------------


    cities = set(key[0] for key in dict_tokens.keys())
    print(f"Cities present: {sorted(cities)}")
    print(f"Total entries: {len(dict_tokens)}")

    df_tdm = build_term_document_matrix(dict_tokens)
    print(f"Term-document matrix shape: {df_tdm.shape}")
    print(df_tdm.head())

    df_tdm = clean_term_document_matrix(df_tdm,None,keep_numbers=False)
    print("After cleaning:")
    print(df_tdm.head())

    df_tdm = normalize_social_media_terms(df_tdm)
    print("After social media normalization:")
    print(df_tdm.head())

    df_tdm = remove_nltk_stopwords(df_tdm)
    print("After NLTK stopwords removal:")
    print(df_tdm.head())

    df_tdm = lemmatize_matrix_nltk(df_tdm)
    print("After lemmatization:")
    print(df_tdm.head())

    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)  
    print("After project stopwords removal:")
    print(df_tdm.head())

    df_tdm = remove_miniwords(df_tdm,min_length=3)
    print("After miniwords removal:")
    print(df_tdm.head())

    df_freq_terms = filter_terms_by_frequency(df_tdm)
    print("After frequency filtering:")
    print(df_freq_terms.head())
    print(f"Final shape: {df_freq_terms.shape}")

    # ---------------------------------
    # CSV preparation
    # ---------------------------------

    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]

    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    df_freq_terms.to_csv(
        "data/processed/df_freq_terms.csv",
        index=False
    )
    print(f"Saved: data/processed/df_freq_terms.csv {df_freq_terms.shape}")

    return df_freq_terms


def compute_tfidf(df_freq_terms):

    terms_only = [c for c in df_freq_terms.columns if c != "city"]

    tfidf_transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True
    )

    X_tfidf = tfidf_transformer.fit_transform(
        df_freq_terms[terms_only].values
    )

    df_tfidf = pd.DataFrame(
        X_tfidf.toarray(),
        columns=terms_only,
        index=df_freq_terms.index
    )

    df_tfidf["city"] = df_freq_terms["city"].values

    print(f"TF-IDF matrix shape: {df_tfidf.shape}")
    print(df_tfidf.head())

    tfidf_by_city = df_tfidf.groupby("city").mean()
    tfidf_by_city_norm = tfidf_by_city.div(
        tfidf_by_city.sum(axis=1),
        axis=0
    )

    print("TF-IDF normalized by city:")
    print(tfidf_by_city_norm.head())

    print("Sum of normalized TF-IDF per city:")
    print(tfidf_by_city_norm.sum(axis=1))

    return df_tfidf, tfidf_by_city_norm



if __name__ == "__main__":
    df_freq_terms = main(tokenize_json_by_city_url("data/processed/corpus_json.json"))
    df_tfidf, tfidf_by_city_norm = compute_tfidf(df_freq_terms)



#get most frequent terms
top_terms = pd.DataFrame(df_freq_terms.drop(columns="city").sum().sort_values(ascending=False).head(20)).reset_index().rename(columns={"index":"term", 0:"frequency"})
print("Top 20 most frequent terms in the corpus:")
print(top_terms)
top_terms = pd.DataFrame(df_freq_terms.drop(columns="city").sum().sort_values(ascending=True).head(20)).reset_index().rename(columns={"index":"term", 0:"frequency"})
print("Top 20 least frequent terms in the corpus:")
print(top_terms)   


short_words = [c for c in df_freq_terms.columns if c != "city" and 1 <= len(c) <= 2]
print(f"Number of short words (1-3 chars): {len(short_words)}")
print("short words:")  
print(short_words)