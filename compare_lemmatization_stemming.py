"""
COMPARISON LEMMATIZATION VS STEMMING
This script is used for methodological comparison.
 The selected method will be integrated in the final pipeline.
"""
from Utils import *
from parameters import *

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

## call the Cleaned term-document matrix
def load_base_matrix_before_morphology(json_path, project_stopwords):
    """
    Build the term-document matrix cleaned up to the point
    just before lemmatization or stemming.

    Input:
        json_path : str
            Path to corpus JSON
        project_stopwords : list or set
            project-specific stopwords

    Output:
        pd.DataFrame
            Cleaned term-document matrix
    """

    dict_tokens = tokenize_json_by_city_url(json_path)
    df_tdm = build_term_document_matrix(dict_tokens)

    df_tdm = clean_term_document_matrix(df_tdm, keep_numbers=False)
    df_tdm = normalize_social_media_terms(df_tdm)
    df_tdm = remove_nltk_stopwords(df_tdm)
    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)
    df_tdm = remove_miniwords(df_tdm, min_length=3)

    return df_tdm

# LOAD CLEAN BASE MATRIX (BEFORE MORPHOLOGY)

df_tdm = load_base_matrix_before_morphology(json_path="data/processed/corpus_json.json",project_stopwords=project_stopwords)
print("Base matrix loaded")
print("Shape:", df_tdm.shape)

# LEMMATIZATION
def lemmatize_matrix_nltk(matrix):
    """
    Apply lemmatization to a term-document matrix
    """
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        lemmatizer = WordNetLemmatizer()

    new_columns = {}

    for term in matrix.columns:
        lemma = lemmatizer.lemmatize(term)

        if lemma in new_columns:
            new_columns[lemma] += matrix[term]
        else:
            new_columns[lemma] = matrix[term].copy()

    return pd.DataFrame(new_columns, index=matrix.index)

# STEMMING
def stem_matrix_nltk(matrix):
    """
    Apply stemming to a term-document matrix 
    """
    stemmer = PorterStemmer()
    new_columns = {}

    for term in matrix.columns:
        stem = stemmer.stem(term)

        if stem in new_columns:
            new_columns[stem] += matrix[term]
        else:
            new_columns[stem] = matrix[term].copy()

    return pd.DataFrame(new_columns, index=matrix.index)

## Evaluation metrics 

def vocabulary_size(matrix):
    return matrix.shape[1]


def matrix_sparsity(matrix):
    return 1.0 - (matrix.astype(bool).sum().sum() / matrix.size)


def top_terms(matrix, n=20):
    return matrix.sum(axis=0).sort_values(ascending=False).head(n)

print("\nApplying lemmatization...")
df_lemma = lemmatize_matrix_nltk(df_tdm)

print("Applying stemming...")
df_stem = stem_matrix_nltk(df_tdm)

# Quantitative metrics
print("\nVOCABULARY SIZE")
print("Lemma:", vocabulary_size(df_lemma))
print("Stem :", vocabulary_size(df_stem))

print("\nMATRIX SPARSITY")
print("Lemma:", round(matrix_sparsity(df_lemma), 4))
print("Stem :", round(matrix_sparsity(df_stem), 4))

# qualitative metrics
print("\nTOP 20 TERMS (LEMMA)")
print(top_terms(df_lemma))

print("\nTOP 20 TERMS (STEM)")
print(top_terms(df_stem))

# recording results
df_lemma.to_csv("data/processed/df_tdm_lemma.csv")
df_stem.to_csv("data/processed/df_tdm_stem.csv")

metrics = {"method": ["lemmatization", "stemming"],"vocabulary_size": [ vocabulary_size(df_lemma),vocabulary_size(df_stem) ],"sparsity": [round(matrix_sparsity(df_lemma), 4),  round(matrix_sparsity(df_stem), 4)  ]}

df_metrics = pd.DataFrame(metrics)

output_path = "data/processed/morphology_comparison.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    df_metrics.to_excel(writer, sheet_name="metrics", index=False)
    top_terms(df_lemma).to_frame("frequency").to_excel(
        writer, sheet_name="top_terms_lemma"
        )
    top_terms(df_stem).to_frame("frequency").to_excel(
        writer, sheet_name="top_terms_stem"
    )

print(f"\nEvaluation metrics saved to {output_path}")
