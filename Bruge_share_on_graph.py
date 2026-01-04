import pandas as pd
import numpy as np


FREQ_PATH = "data/processed/df_freq_terms.csv"          

NODES_JACCARD_PATH = "gephi_export/nodes_jaccard.csv"  

OUTPUT_NODES_PATH = "gephi_export/nodes_jaccard_brugeShare.csv"


def compute_bruges_share_per_term(freq_path: str,
                                  city_name: str = "Bruges") -> pd.Series:
    """
    Calculates for each term the part of the total frequency that comes from the city city_name:
    share(w) = freq_city(w) / freq_total(w)
    """
    df = pd.read_csv(freq_path)

    if "city" not in df.columns:
        raise ValueError("Column 'city' is required in df_freq_terms.csv")

    term_cols = [c for c in df.columns if c != "city"]

    # fréquency per city
    freq_by_city = df.groupby("city")[term_cols].sum()

    if city_name not in freq_by_city.index:
        raise ValueError(f"City'{city_name}' not found in df_freq_terms.csv")

    brugge_vec = freq_by_city.loc[city_name]
    total_vec = freq_by_city.sum(axis=0)

    share = (brugge_vec / total_vec).fillna(0)

    return share


def add_bruges_share_to_nodes(nodes_path: str,
                              share: pd.Series,
                              output_path: str):
    """
    Adds a 'bruges_share' column to the Gephi (Jaccard) node file.
    Terms missing in 'share' receive 0.
    """
    df_nodes = pd.read_csv(nodes_path)

    if "Id" in df_nodes.columns:
        id_col = "Id"
    elif "id" in df_nodes.columns:
        id_col = "id"
    else:
        raise ValueError("Column'Id' not found in nodes_jaccard.csv")

    # map term -> share of Bruges
    df_nodes["bruges_share"] = df_nodes[id_col].map(share).fillna(0.0)

    df_nodes.to_csv(output_path, index=False)
    print(f"Updated nodes file with 'bruges_share' saved to {output_path}")


def main():
    # 1. calculate the Bruges share per term
    share = compute_bruges_share_per_term(FREQ_PATH, city_name="Bruges")

    # 2. add it to the Jaccard node file
    add_bruges_share_to_nodes(
        NODES_JACCARD_PATH,
        share,
        OUTPUT_NODES_PATH
    )

if __name__ == "__main__":
    main()
