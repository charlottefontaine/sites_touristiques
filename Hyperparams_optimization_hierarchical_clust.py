# use to determine which parameters are efficient for the hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

BASE_PATH = "data/processed"
TFIDF_CITY_PATH = os.path.join(BASE_PATH, "tfidf_by_city_norm.csv")
OUTPUT_DIR = "data/city_similarity"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_city_dendrogram(df_tfidf_city: pd.DataFrame,
                         metric: str = "cosine",
                         method: str = "average"):
    """
    Perform hierarchical clustering and plot a dendrogram of cities
    with configurable distance metric and linkage method.
    """
    # 1) distance matrix
    dist_vec = pdist(df_tfidf_city.values, metric=metric)  

    # 2) hierarchic linkage
    linkage_matrix = linkage(dist_vec, method=method)  

    # 3) plot
    plt.figure(figsize=(8, 4))
    dendrogram(linkage_matrix,
               labels=df_tfidf_city.index.tolist(),
               leaf_rotation=45)
    plt.title(f"Hierarchical Clustering of Cities\nmetric={metric}, linkage={method}")
    plt.ylabel("Distance")
    plt.tight_layout()

    filename = f"city_dendrogram_{metric}_{method}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.show()
    print(f"Dendrogram saved to: {path}")


def load_tfidf_by_city(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df

def main():
    if not os.path.exists(TFIDF_CITY_PATH):
        print(f"Error: {TFIDF_CITY_PATH} not found.")
        return

    df_tfidf_city = load_tfidf_by_city(TFIDF_CITY_PATH)
    print("TF-IDF by city shape:", df_tfidf_city.shape)

    # 3. Tests
    metrics = ["cosine", "euclidean","cityblock"]
    methods = ["single", "complete", "average"]

    for metric in metrics:
        for method in methods:
            print(f"\n== Dendrogram with metric={metric}, method={method} ==")
            plot_city_dendrogram(df_tfidf_city, metric=metric, method=method)

    # Ward with euclidean (the only possible distance metric)
    print("\n== Dendrogram with metric=euclidean, method=ward ==")
    plot_city_dendrogram(df_tfidf_city, metric="euclidean", method="ward")


if __name__ == "__main__":
    main()
