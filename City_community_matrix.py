"""
city_community_weights_jaccard.py

Builds a city–community matrix from the global Jaccard co-occurrence graph.

The script:
- loads the global word co-occurrence graph (Jaccard) exported for Gephi,
- loads the term-frequency matrix with a 'city' column (df_freq_terms.csv),
- aggregates term frequencies by city,
- computes, for each city and each word community, the share of frequency
  devoted to that community (row-normalized),
- exports a city x community weight matrix as CSV.

Inputs
------
- data/link_analysis/coocurence_window_graph/nodes_jaccard.csv
- data/link_analysis/coocurence_window_graph/edges_jaccard.csv
    (Gephi exports from the sliding-window co-occurrence script)
- data/processed/df_freq_terms.csv
    Document-term frequency matrix with a 'city' column.

Outputs
-------
- data/link_analysis/coocurence_window_graph/city_community_weights_jaccard.csv
    Rows = cities, columns = Community_k, values = normalized weights per city.
"""
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import os

def load_graph_jaccard(nodes_path: str, edges_path: str) -> nx.Graph:
    """
    Loads the Jaccard graph exported for Gephi:
    - nodes_jaccard.csv  
    - edges_jaccard.csv
    """
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)

    G = nx.Graph()
    # nodes
    for _, row in df_nodes.iterrows():
        G.add_node(row["Id"])

    # weighted hedges
    for _, row in df_edges.iterrows():
        G.add_edge(row["Source"], row["Target"],
                   weight=row.get("Weight", 1.0))

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_freq_matrix(csv_path: str) -> pd.DataFrame:
    """
    Loads df_freq_terms.csv (documents x terms) with a 'city' column.
    """
    df = pd.read_csv(csv_path)
    if "city" not in df.columns:
        raise ValueError("The 'city' column is required in df_freq_terms.csv")
    return df


def compute_city_term_frequencies(df_freq: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates term frequencies by city.
    Returns a DataFrame (cities x terms).
    """
    term_cols = [c for c in df_freq.columns if c != "city"]
    df_city = df_freq.groupby("city")[term_cols].sum()
    return df_city

def compute_city_community_matrix(G: nx.Graph,
                                  df_city_terms: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate, for each city  the total weight devoted to each community of words.
    - G: co-occurrence graph (nodes = terms)
    - df_city_terms: DataFrame (cities x terms) with frequencies
    Return: DataFrame (cities x communities) with normalized scores per city.
    """
    # 1. Word communities on the Jaccard graph
    communities = list(greedy_modularity_communities(G))
    print(f"{len(communities)} communities detected.")

    cities = df_city_terms.index.tolist()
    n_cities = len(cities)
    n_comms = len(communities)

    M = np.zeros((n_cities, n_comms), dtype=float)

    available_terms = set(df_city_terms.columns)

    for k, comm in enumerate(communities):
        comm_terms = [t for t in comm if t in available_terms]
        if not comm_terms:
            continue
        # sum of frequencies of all community terms by city
        M[:, k] = df_city_terms[comm_terms].sum(axis=1).values

    # 2. Normalization by city (each row = 1)
    M_norm = M / M.sum(axis=1, keepdims=True)
    M_norm = np.nan_to_num(M_norm)  # replace NaN (lines at 0) with 0

    col_names = [f"Community_{k}" for k in range(n_comms)]
    df_city_comm = pd.DataFrame(M_norm, index=cities, columns=col_names)

    return df_city_comm


def main():
    nodes_path = "data/link_analysis/coocurence_window_graph/nodes_jaccard.csv"
    edges_path = "data/link_analysis/coocurence_window_graph/edges_jaccard.csv"
    G = load_graph_jaccard(nodes_path, edges_path)

    df_freq = load_freq_matrix("data/processed/df_freq_terms.csv")
    df_city_terms = compute_city_term_frequencies(df_freq)

    df_city_comm = compute_city_community_matrix(G, df_city_terms)

    output_path = "data/link_analysis/coocurence_window_graph/city_community_weights_jaccard.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_city_comm.to_csv(output_path, index=True)

    print("City–community matrix saved to:")
    print(os.path.abspath(output_path))



if __name__ == "__main__":
    main()
