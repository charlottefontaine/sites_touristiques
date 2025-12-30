import os
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# topic_modeling : Latent Dirichlet Allocation

BASE_PATH = "data/processed"
FREQ_TDM_PATH = os.path.join(BASE_PATH, "df_freq_terms.csv")
OUTPUT_DIR = "data/topics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping topic id → English category name
TOPIC_LABELS = {
    0: "Romantic City Walks & Heritage",
    1: "Transportation & Guided Sightseeing",
    2: "Cultural Discovery & Museum Passes",
    3: "Food, Wine & Local Experiences",
    4: "Tourist Information & Digital Platforms",
    5: "Entertainment, Shopping & Family Attractions"
}

def load_term_document_matrix(csv_path: str):
    df = pd.read_csv(csv_path)
    if "city" not in df.columns:
        raise ValueError("Column 'city' not found in df_freq_terms.csv")
    cities = df["city"].tolist()
    terms = [c for c in df.columns if c != "city"]
    tdm = df[terms].values
    return tdm, terms, np.array(cities)

def fit_lda(tdm: np.ndarray, n_topics: int = 8, max_iter: int = 100, random_state: int = 42):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="batch",
        random_state=random_state,
        n_jobs=-1
    )
    doc_topic = lda.fit_transform(tdm)
    return lda, doc_topic

def evaluate_lda_models(tdm, n_topics_list, max_iter=100):
    results = []
    for k in n_topics_list:
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=max_iter,
            learning_method="batch",
            random_state=42,
            n_jobs=-1
        )
        doc_topic = lda.fit_transform(tdm)
        loglik = lda.score(tdm)
        perp = lda.perplexity(tdm)
        results.append({"k": k, "score": loglik, "perplexity": perp})
    return pd.DataFrame(results)

def print_top_words_per_topic(lda_model, terms, n_top_words: int = 15 ):
    topics_dict = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_terms = [terms[i] for i in top_indices]
        topics_dict[topic_idx] = top_terms
        topic_name = TOPIC_LABELS.get(topic_idx, f"Topic {topic_idx}")
        print(f"\nTopic {topic_idx} — {topic_name}:")
        print(", ".join(top_terms))
    return topics_dict

def aggregate_topics_by_city(doc_topic: np.ndarray, cities: np.ndarray):
    df_doc_topics = pd.DataFrame(doc_topic)
    df_doc_topics = df_doc_topics.rename(columns=TOPIC_LABELS)
    df_doc_topics["city"] = cities
    df_city_topics = df_doc_topics.groupby("city").mean()
    df_city_topics = df_city_topics.div(df_city_topics.sum(axis=1), axis=0)
    return df_city_topics

def plot_city_topic_heatmap(df_city_topics: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_city_topics, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Topic Proportions by City")
    plt.xlabel("Topic")
    plt.ylabel("City")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "city_topic_heatmap.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Heatmap saved to: {save_path}")

def main():
    if not os.path.exists(FREQ_TDM_PATH):
        print(f"Error: {FREQ_TDM_PATH} not found.")
        return

    tdm, terms, cities = load_term_document_matrix(FREQ_TDM_PATH)
    print(f"TDM shape for LDA: {tdm.shape}")

    metrics = evaluate_lda_models(tdm, n_topics_list=[2,3,4,5,6,7,8,9,10])
    print("\nCompare models:")
    print(metrics)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(metrics["k"], metrics["score"], "o-")
    plt.xlabel("Number of topics")
    plt.ylabel("Log-likelihood")
    plt.title("Score per number of topics")
    plt.subplot(1,2,2)
    plt.plot(metrics["k"], metrics["perplexity"], "o-")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.title("Perplexity per number of topics")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lda_metrics.png"))
    plt.show()

    n_topics = 6
    lda_model, doc_topic = fit_lda(tdm, n_topics=n_topics)
    topics_dict = print_top_words_per_topic(lda_model, terms, n_top_words=15)

    df_city_topics = aggregate_topics_by_city(doc_topic, cities)
    print("\nTopic proportions by city (rows sum to 1):")
    print(df_city_topics.round(3))

    df_city_topics.to_csv(os.path.join(OUTPUT_DIR, "city_topic_distribution.csv"))
    print(f"City-topic distribution saved to {OUTPUT_DIR}/city_topic_distribution.csv")

    plot_city_topic_heatmap(df_city_topics, OUTPUT_DIR)

if __name__ == "__main__":
    main()
