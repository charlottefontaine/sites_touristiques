from Utils import *
from parameters import *

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

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

    # Ajout colonne city
    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    os.makedirs("data/processed", exist_ok=True)
    return df_freq_terms


# ---------------------------------
# Part 2 . Heatmap Excel
# ---------------------------------

def save_heatmap_excel(df, output_path, min_count=4, max_words=100):
    # Colonnes numériques uniquement
    df_numeric = df.drop(columns=["city"]).apply(pd.to_numeric, errors="coerce").fillna(0)

    # Fréquences du corpus de CETTE heatmap
    corpus_freq = df_numeric.sum(axis=0)

    # Mots dits au moins min_count fois
    corpus_freq = corpus_freq[corpus_freq >= min_count]

    # Top max_words mots les plus fréquents
    top_terms = corpus_freq.sort_values(ascending=False).head(max_words).index
    df_numeric = df_numeric[top_terms]

    # Reconstruction finale
    df_to_save = pd.concat([df["city"], df_numeric], axis=1)
    df_to_save.to_excel(output_path, index=False)

    # --- Mise en forme heatmap Excel ---
    wb = load_workbook(output_path)
    ws = wb.active

    max_val = df_numeric.to_numpy().max()

    for row in ws.iter_rows(
        min_row=2,
        max_row=ws.max_row,
        min_col=2,
        max_col=ws.max_column
    ):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                intensity = int(255 * cell.value / max_val) if max_val != 0 else 0
                red = 255
                green = 255 - intensity
                blue = 255 - intensity
                cell.fill = PatternFill(
                    start_color=f"{red:02X}{green:02X}{blue:02X}",
                    end_color=f"{red:02X}{green:02X}{blue:02X}",
                    fill_type="solid"
                )

    wb.save(output_path)


# ---------------------------------
# Main processing (PAR ZONE)
# ---------------------------------

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

    # Sauvegarde TDM agrégé
    df_terms_only = df_freq_terms.drop(columns=["city"])
    df_zone = pd.DataFrame(df_terms_only.sum()).T
    output_csv = f"data/processed/tdm_{zone}.csv"
    df_zone.to_csv(output_csv, index=False)
    print(f"TDM saved: {output_csv}")

    # 🔥 Heatmap PAR ZONE
    heatmap_path = f"data/processed/heatmap_{zone}.xlsx"
    save_heatmap_excel(df_freq_terms, heatmap_path)
    print(f"Heatmap saved: {heatmap_path}")


# ---------------------------------
# Global corpus (optionnel – LDA)
# ---------------------------------

if __name__ == "__main__":
    dict_tokens_full = tokenize_json_by_city_url("data/processed/corpus_json.json")
    df_freq_terms_full = main(dict_tokens_full)
    df_freq_terms_full.to_csv(
        "data/processed/df_freq_terms.csv",
        index=False
    )
    print("Saved global frequency matrix: data/processed/df_freq_terms.csv")
