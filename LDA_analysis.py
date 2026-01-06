"""
Creates an LDA topic modeling on the df_freq_terms.csv matrix:
- evaluates several values ​​of k (2 to 10) via log-likelihood and perplexity,
- inspects the top words per topic for a few k,
- trains a final model (k = 6) and aggregates the distributions of topics by city,
- saves the results and a heatmap.

Input
-----
data/processed/df_freq_terms.csv
    Documents x terms matrix, with a 'city' column.

Outputs
-------
data/text_analysis/lda/lda_metrics.png
    Scores (log-likelihood, perplexity) as a function of k.

data/text_analysis/lda/city_topic_distribution.csv
    Cities x topics matrix (rows normalized to 1).

data/text_analysis/lda/city_topic_heatmap.png
    Heatmap of topic proportions by city.
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# topic_modeling : Latent Dirichlet Allocation
BASE_DIR = "data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
LDA_DIR = os.path.join(BASE_DIR, "text_analysis", "lda")

FREQ_TDM_PATH = os.path.join(PROCESSED_DIR, "df_freq_terms.csv")
os.makedirs(LDA_DIR, exist_ok=True)

def load_term_document_matrix(csv_path: str):
    """
    Load the cleaned term-document matrix.

    Input:
        csv_path: path to df_freq_terms.csv (first column = city, others = terms)
    Output:
        tdm: 2D numpy array (documents x terms)
        terms: list of term names 
        cities: list of city names 
    """
    df = pd.read_csv(csv_path)

    if "city" not in df.columns:
        raise ValueError("Column 'city' not found in df_freq_terms.csv")

    cities = df["city"].tolist()
    terms = [c for c in df.columns if c != "city"]
    tdm = df[terms].values

    return tdm, terms, np.array(cities)


def fit_lda(tdm: np.ndarray, n_topics: int = 8, max_iter: int = 100, random_state: int = 42):
    """
    Fit LDA model on the term-document matrix.

    Input:
        tdm: document-term matrix 
        n_topics: number of latent topics to extract
        max_iter: number of EM iterations
        random_state: seed for reproducibility
    Output:
        lda_model: fitted LatentDirichletAllocation object
        doc_topic: matrix (documents x topics) with topic proportions
    """
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
    """
    Evaluate LDA models with a different number of topics 
    Return DataFrame with score (log-likelihood) and perplexity.
    """
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
    """
    Print top words for each topic 

    Input
        lda_model: fitted LDA
        terms: list of vocabulary terms
        n_top_words: number of words to display per topic
    Output:
        topics_dict: dict {topic_id: [list of top words]}
    """
    topics_dict = {}

    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_terms = [terms[i] for i in top_indices]
        topics_dict[topic_idx] = top_terms

        print(f"\nTopic {topic_idx}:")
        print(", ".join(top_terms))

    return topics_dict


def aggregate_topics_by_city(doc_topic: np.ndarray, cities: np.ndarray):
    """
    Aggregate document-topic distributions at city level.

    Input:
        doc_topic: matrix (documents x topics)
        cities: array of city names aligned with documents
    Output:
        df_city_topics: DataFrame (cities x topics) with normalized topic proportions
    """
    df_doc_topics = pd.DataFrame(doc_topic)
    df_doc_topics["city"] = cities

    # Average topic proportion per city
    df_city_topics = df_doc_topics.groupby("city").mean()

    # Normalize rows so each city sums to 1
    df_city_topics = df_city_topics.div(df_city_topics.sum(axis=1), axis=0)

    return df_city_topics


def plot_city_topic_heatmap(df_city_topics: pd.DataFrame, output_dir: str):
    """
    Plot a heatmap of topic proportions per city.

    Input:
        df_city_topics: DataFrame (cities x topics)
    Output:
        saves PNG
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_city_topics, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Topic Proportions by City")
    plt.xlabel("Topic")
    plt.ylabel("City")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "city_topic_heatmap.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Heatmap saved to: {save_path}")


def main():
    #  Load cleaned TDM
    if not os.path.exists(FREQ_TDM_PATH):
        print(f"Error: {FREQ_TDM_PATH} not found.")
        return

    tdm, terms, cities = load_term_document_matrix(FREQ_TDM_PATH)
    print(f"TDM shape for LDA: {tdm.shape}")

    # Step 1 : evaluate different number of topics
    metrics = evaluate_lda_models(tdm, n_topics_list=[2,3,4,5, 6, 7,8,9, 10])
    print("\nCompare models :")
    print(metrics)

    # Show the improvement in function of the number of topics
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

    # 2) Validate k with top words
    for k in [4, 6, 8]:
      print(f"\n=== Investigate topics for k = {k} ===")
      lda_k, doc_topic_k = fit_lda(tdm, n_topics=k)
      topics_dict_k = print_top_words_per_topic(lda_k, terms, n_top_words=15)

    # Step 2 : Train the selected model
    # Fit LDA
    n_topics = 6 
    lda_model, doc_topic = fit_lda(tdm, n_topics=n_topics)

    # topics and words
    topics_dict = print_top_words_per_topic(lda_model, terms, n_top_words=15)

    # Aggregate at city level
    df_city_topics = aggregate_topics_by_city(doc_topic, cities)
    print("\nTopic proportions by city (rows sum to 1):")
    print(df_city_topics.round(3))

    #  Save results
    df_city_topics.to_csv(os.path.join(OUTPUT_DIR, "city_topic_distribution.csv"))
    print(f"City-topic distribution saved to {OUTPUT_DIR}/city_topic_distribution.csv")

    # Visualize
    plot_city_topic_heatmap(df_city_topics, OUTPUT_DIR)
 

if __name__ == "__main__":
    main()