import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the cleaned term-document matrix
df = pd.read_csv('df_freq_terms.csv')

# ===============================
# Select top 500 most frequent terms
# ===============================
term_frequencies = df.sum(axis=0)

TOP_N = 500
top_terms = term_frequencies.sort_values(ascending=False).head(TOP_N).index

df_top = df[top_terms]

X_words = df_top.T.values
terms = df_top.columns.tolist()


# No need to select top terms since df_freq_terms is already filtered

N_CLUSTERS = 6

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

kmeans.fit(X_words)
labels = kmeans.labels_

clusters_df = pd.DataFrame({
    "term": terms,
    "cluster": labels
})


for c in range(N_CLUSTERS):
    print(f"\nCluster {c}")
    print(
        clusters_df[clusters_df["cluster"] == c]
        .head(15)["term"]
        .tolist()
    )


# ===============================
# 7. OPTIONAL – Manual thematic labeling
# ===============================
cluster_labels = {
    0: "Culture",
    1: "Gastronomy",
    2: "Events",
    3: "Well-being",
    4: "Economy",
    5: "Other"
}

clusters_df["theme"] = clusters_df["cluster"].map(cluster_labels)



# ===============================
# 8. Compute thematic scores per document
# ===============================
tfidf_df = df  # Use the cleaned frequency matrix directly

term_to_theme = clusters_df.set_index("term")["theme"]

theme_scores = {}

for theme in cluster_labels.values():
    theme_terms = term_to_theme[term_to_theme == theme].index
    theme_scores[theme] = tfidf_df[theme_terms].sum(axis=1)

theme_scores_df = pd.DataFrame(theme_scores)

print("\nThematic scores per document:")
print(theme_scores_df.head())


# ===============================
# 9. Aggregate thematic scores by city
# ===============================

# Join city information
theme_scores_city = (
    theme_scores_df
    .join(df["city"])
    .groupby("city")
    .mean()
)

print("\nAverage thematic scores per city:")
print(theme_scores_city)

# ===============================
# 9. Visualization (PCA)
# ===============================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_words)

plt.figure(figsize=(10, 8))
for c in range(N_CLUSTERS):
    idx = labels == c
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Cluster {c}",
        alpha=0.6
    )

plt.legend()
plt.title("Word clusters based on cleaned frequency matrix")
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.show()


# ===============================
# 10. Save results
# ===============================
clusters_df.to_csv('word_clusters.csv', index=False)

print("\nClustering results saved to 'word_clusters.csv'")
