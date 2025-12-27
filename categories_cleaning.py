from corpus_cleaning import main, compute_tfidf
from Utils import tokenize_json_by_city_url
from parameters import *
import pandas as pd

# corpus preparation
dict_tokens = tokenize_json_by_city_url("data/processed/corpus_json.json")
df_freq_terms = main(dict_tokens)
df_tfidf, tfidf_by_city_norm = compute_tfidf(df_freq_terms)

# categories division
city_groups = {
    "sea": SEA_CITIES,
    "nosea": NOSEA_CITIES,
    "south": SOUTH_CITIES,
    "north": NORTH_CITIES
}

# Save TDMs for each group
for group_name, cities in city_groups.items():
    df_group = df_tfidf[df_tfidf["city"].isin(cities)].copy()
    df_group.to_csv(f"data/processed/tdm_{group_name}.csv", index=False)
    print(f"Saved {group_name} TDM: {df_group.shape}")

print("Category-specific TDMs saved.")
