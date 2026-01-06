"""
Calculates the similarity between cities from their TF-IDF profiles
and performs hierarchical clustering.

Input
-----
data/processed/tfidf_by_city_norm.csv
    Cities x terms matrix (rows normalized to 1).

Outputs
-------
data/text_analysis/city_similarity/city_similarity_matrix.csv
    Cosine similarity matrix between cities.

data/text_analysis/city_similarity/city_similarity_heatmap.png
    Heatmap of cosine similarities.

data/text_analysis/city_similarity/city_dendrogram.png
    Dendrogram of hierarchical clustering (Ward) of cities.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TFIDF_CITY_PATH = os.path.join(PROCESSED_DIR, "tfidf_by_city_norm.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "text_analysis", "city_similarity")
os.makedirs(OUTPUT_DIR, exist_ok=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tfidf_by_city(path: str) -> pd.DataFrame:
    """
    Load the city-level TF-IDF matrix.
    """
    df = pd.read_csv(path, index_col=0)
    return df

def compute_similarity_matrix(df_tfidf_city: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between cities based on TF-IDF profiles.

    Output:
        DataFrame (cities x cities) with cosine similarity.
    """
    sim = cosine_similarity(df_tfidf_city.values)
    cities = df_tfidf_city.index.tolist()
    df_sim = pd.DataFrame(sim, index=cities, columns=cities)
    return df_sim

def plot_similarity_heatmap(df_sim: pd.DataFrame):
    """
    Plot and save a heatmap of city similarities.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Cosine Similarity Between City Profiles (TF-IDF)")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "city_similarity_heatmap.png")
    plt.savefig(path)
    plt.show()
    print(f"Heatmap saved to: {path}")

def plot_city_dendrogram(df_tfidf_city: pd.DataFrame):
    """
    Perform hierarchical clustering and plot a dendrogram of cities.
    """
    linkage_matrix = linkage(df_tfidf_city.values, method="ward")
    plt.figure(figsize=(8, 4))
    dendrogram(linkage_matrix, labels=df_tfidf_city.index.tolist(), leaf_rotation=45)
    plt.title("Hierarchical Clustering of Cities (TF-IDF Profiles)")
    plt.ylabel("Distance")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "city_dendrogram.png")
    plt.savefig(path)
    plt.show()
    print(f"Dendrogram saved to: {path}")

def main():
    if not os.path.exists(TFIDF_CITY_PATH):
        print(f"Error: {TFIDF_CITY_PATH} not found.")
        return

    df_tfidf_city = load_tfidf_by_city(TFIDF_CITY_PATH)
    print("TF-IDF by city shape:", df_tfidf_city.shape)

    # 1. Similarity matrix
    df_sim = compute_similarity_matrix(df_tfidf_city)
    print("\nCosine similarity between cities:")
    print(df_sim.round(3))

    sim_csv = os.path.join(OUTPUT_DIR, "city_similarity_matrix.csv")
    df_sim.to_csv(sim_csv)
    print(f"Similarity matrix saved to: {sim_csv}")

    # 2. Visualizations
    plot_similarity_heatmap(df_sim)
    plot_city_dendrogram(df_tfidf_city)

if __name__ == "__main__":
    main()
