# file: link_analysis.py

import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import networkx as nx

from Utils import extract_links_from_html, filter_technical_links
from parameters import CITIES, REQUEST_TIMEOUT  

BASE_OUTPUT = "data"
HTML_HOMEPAGES_DIR = os.path.join(BASE_OUTPUT, "html", "homepages")
HTML_DEPTH1_DIR = os.path.join(BASE_OUTPUT, "html", "depth1")
PROCESSED_DIR = os.path.join(BASE_OUTPUT, "processed")

LINKS_CSV = os.path.join(PROCESSED_DIR, "links.csv")
GRAPH_METRICS_CSV = os.path.join(PROCESSED_DIR, "graph_metrics.csv")
GEPHI_EXPORT_DIR = os.path.join(PROCESSED_DIR, "gephi")


def degree_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    """
    Matrice des degrés (in ou out) pour un graphe dirigé.
    A[i,j] = 1 s'il y a un lien i -> j.
    """
    if direction == "out":
        deg = A.sum(axis=1)  # rows = inbound links
    elif direction == "in":
        deg = A.sum(axis=0)  # columns = outbound links
    else:
        raise ValueError("direction must be 'out' or 'in'")
    return np.diag(deg)


def transition_matrix(A: np.ndarray, direction: str = "out") -> np.ndarray:
    """
    transition matrix P 
    P[i,j] = probability of going from i to j.
    """
    D = degree_matrix(A, direction)
    deg = np.diag(D)
    P = np.zeros_like(A, dtype=float)
    for i, d in enumerate(deg):
        if d > 0:
            P[i, :] = A[i, :] / d
    return P


def pagerank_power_iteration(A: np.ndarray,
                             alpha: float = 0.85,
                             max_iter: int = 100,
                             tol: float = 1e-8) -> np.ndarray:
    """.
    PR_{k+1} = alpha * P^T * PR_k + (1 - alpha)/n * 1
    """
    n = A.shape[0]
    if n == 0:
        return np.array([])

    P = transition_matrix(A, direction="out")
    u = np.ones(n) / n
    pr = np.ones(n) / n

    for _ in range(max_iter):
        pr_new = alpha * (P.T @ pr) + (1 - alpha) * u
        pr_new = pr_new / pr_new.sum()
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new

    return pr


def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    """
    min distance
    0 = no hedge, 1 = hedge; no reachable -> 100000.
    unweighted graph
    """
    SP = A.copy().astype(float)
    n = SP.shape[0]

    for i in range(n):
        for j in range(n):
            if i == j:
                SP[i, j] = 0.0
            elif SP[i, j] == 0:
                SP[i, j] = 100000.0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if SP[i, j] > SP[i, k] + SP[k, j]:
                    SP[i, j] = SP[i, k] + SP[k, j]

    return SP


def closeness_centrality(A: np.ndarray) -> np.ndarray:
    """
        C(i) = (n-1) / sum_j d(i,j)
        d(i,j) from shortest_path_matrix.
    """
    SP = shortest_path_matrix(A)
    n = SP.shape[0]
    cc = np.zeros(n, dtype=float)

    for i in range(n):
        d = SP[i, :]
        mask = (d < 100000) & (np.arange(n) != i)
        if mask.sum() == 0:
            cc[i] = 0.0
        else:
            cc[i] = (mask.sum()) / d[mask].sum()
    return cc


# Build graph

def load_html_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_links_dataframe() -> pd.DataFrame:
    rows = []

    # 1. Homepages (depth 0)
    for city, homepage_url in CITIES.items():
        homepage_path = os.path.join(HTML_HOMEPAGES_DIR, f"{city}.html")
        if not os.path.exists(homepage_path):
            continue

        html = load_html_file(homepage_path)
        all_links = extract_links_from_html(html, homepage_url)
        domain = urlparse(homepage_url).netloc
        depth1_links = filter_technical_links(all_links, domain)

        for target in depth1_links:
            rows.append({
                "city": city,
                "source_url": homepage_url,
                "target_url": target
            })

    # 2. depth 1 page
    corpus_path = os.path.join(PROCESSED_DIR, "corpus.csv")
    df_corpus = pd.read_csv(corpus_path)

    url_by_city = {
        (row["city"], hash(row["url"])): row["url"]
        for _, row in df_corpus.iterrows()
    }

    for fname in os.listdir(HTML_DEPTH1_DIR):
        if not fname.endswith(".html"):
            continue

        city_part, hash_part = fname.split("_", 1)
        city = city_part.strip()
        hash_str = os.path.splitext(hash_part)[0]

        try:
            h = int(hash_str)
        except ValueError:
            continue

        source_url = url_by_city.get((city, h))
        if source_url is None:
            continue

        html_path = os.path.join(HTML_DEPTH1_DIR, fname)
        html = load_html_file(html_path)

        all_links = extract_links_from_html(html, source_url)
        domain = urlparse(source_url).netloc
        internal_links = filter_technical_links(all_links, domain)

        for target in internal_links:
            rows.append({
                "city": city,
                "source_url": source_url,
                "target_url": target
            })

    df_links = pd.DataFrame(rows).drop_duplicates()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_links.to_csv(LINKS_CSV, index=False)
    print(f"Links saved to {LINKS_CSV}  (rows={len(df_links)})")

    return df_links


def build_city_graph(df_links: pd.DataFrame, city: str) -> nx.DiGraph:
    sub = df_links[df_links["city"] == city]
    G = nx.DiGraph()
    for _, row in sub.iterrows():
        src = row["source_url"]
        tgt = row["target_url"]
        G.add_edge(src, tgt)
    return G


def adjacency_matrix_from_graph(G: nx.DiGraph):
    """
    Adjacency matrix + list of nodes
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)
    for src, tgt in G.edges():
        i = idx[src]
        j = idx[tgt]
        A[i, j] = 1
    return A, nodes

# Compute metrics

def compute_graph_metrics(df_links: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for city in sorted(df_links["city"].unique()):
        print(f"Processing graph metrics for {city}")
        G = build_city_graph(df_links, city)
        if G.number_of_nodes() == 0:
            continue

        A, nodes = adjacency_matrix_from_graph(G)

        # degree in/out
        D_out = degree_matrix(A, "out")
        D_in = degree_matrix(A, "in")
        out_deg = np.diag(D_out)
        in_deg = np.diag(D_in)

        # PageRank 
        pr_vec = pagerank_power_iteration(A, alpha=0.85, max_iter=200)

        # Closeness
        cc_vec = closeness_centrality(A)

        for i, url in enumerate(nodes):
            all_rows.append({
                "city": city,
                "url": url,
                "in_degree": int(in_deg[i]),
                "out_degree": int(out_deg[i]),
                "pagerank": float(pr_vec[i]),
                "closeness": float(cc_vec[i])
            })

    df_metrics = pd.DataFrame(all_rows)
    df_metrics.to_csv(GRAPH_METRICS_CSV, index=False)
    print(f"Graph metrics saved to {GRAPH_METRICS_CSV}  (rows={len(df_metrics)})")

    return df_metrics

#  Gephi export
def export_gephi_files(df_links: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    """
    2 CSV :
    - nodes_{city}.csv  (id, label, + métriques)
    - edges_{city}.csv  (source, target)
    """
    os.makedirs(GEPHI_EXPORT_DIR, exist_ok=True)

    for city in sorted(df_links["city"].unique()):
        sub_links = df_links[df_links["city"] == city].copy()
        if sub_links.empty:
            continue

        nodes = pd.unique(
            sub_links[["source_url", "target_url"]].values.ravel()
        )
        df_nodes = pd.DataFrame({"id": nodes, "label": nodes})

        sub_metrics = df_metrics[df_metrics["city"] == city].copy()
        df_nodes = df_nodes.merge(
            sub_metrics.drop(columns=["city"]),
            left_on="id",
            right_on="url",
            how="left"
        ).drop(columns=["url"])

        nodes_path = os.path.join(GEPHI_EXPORT_DIR, f"nodes_{city}.csv")
        df_nodes.to_csv(nodes_path, index=False)

        df_edges = sub_links.rename(
            columns={"source_url": "source", "target_url": "target"}
        )[["source", "target"]]
        edges_path = os.path.join(GEPHI_EXPORT_DIR, f"edges_{city}.csv")
        df_edges.to_csv(edges_path, index=False)

        print(f"Gephi files for {city} saved to:")
        print(f"  {nodes_path}")
        print(f"  {edges_path}")

# Main
def main():
    if os.path.exists(LINKS_CSV):
        df_links = pd.read_csv(LINKS_CSV)
        print(f"Loaded existing links from {LINKS_CSV}")
    else:
        df_links = build_links_dataframe()

    df_metrics = compute_graph_metrics(df_links)
    export_gephi_files(df_links, df_metrics)

if __name__ == "__main__":
    main()
