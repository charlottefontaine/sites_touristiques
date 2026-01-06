"""
Constructs TDM matrices specific to each category of cities
(sea / nosea / north / south) from the TF-IDF matrix by city.
Input
-----
data/processed/tfidf_by_city_norm.csv
    Average TF-IDF matrix per city (rows = cities).

Outputs
-------
data/processed/tdm_sea.csv
data/processed/tdm_nosea.csv
data/processed/tdm_south.csv
data/processed/tdm_north.csv
    Submatrix filtered by city group.
note : The produced matrices (tdm_sea.csv, etc.) contain TF-IDF average weights per city (already standardized) 
"""
from corpus_cleaning import main, compute_tfidf
from Utils import tokenize_json_by_city_url
from parameters import *
import pandas as pd
import os

BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    tfidf_path = os.path.join(PROCESSED_DIR, "tfidf_by_city_norm.csv")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(
            f"{tfidf_path} not found."
        )
    
    df_tfidf = pd.read_csv(tfidf_path)
    if "city" not in df_tfidf.columns:
        raise ValueError("Expected 'city' column in tfidf_by_city_norm.csv")

    city_groups = {
        "sea": SEA_CITIES,
        "nosea": NOSEA_CITIES,
        "south": SOUTH_CITIES,
        "north": NORTH_CITIES,
    }

    for group_name, cities in city_groups.items():
        df_group = df_tfidf[df_tfidf["city"].isin(cities)].copy()
        out_path = os.path.join(PROCESSED_DIR, f"tdm_{group_name}.csv")
        df_group.to_csv(out_path, index=False)
        print(f"Saved {group_name} TDM: {df_group.shape} -> {out_path}")

    print("Category-specific TDMs saved.")

if __name__ == "__main__":
    main()