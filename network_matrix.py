import pandas as pd
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TFIDF = os.path.join(BASE_DIR, "data", "processed", "df_tfidf_by_city.csv")
OUTPUT_NODES = os.path.join(BASE_DIR, "data", "processed", "gephi_nodes.csv")
OUTPUT_EDGES = os.path.join(BASE_DIR, "data", "processed", "gephi_edges.csv")

# Topics LDA pour la correspondance
TOPIC_LABELS = {
    0: "Romantic City Walks & Heritage",
    1: "Transportation & Guided Sightseeing",
    2: "Cultural Discovery & Museum Passes",
    3: "Food, Wine & Local Experiences",
    4: "Tourist Information & Digital Platforms",
    5: "Entertainment, Shopping & Family Attractions"
}


# Enrichis ces listes avec les mots affichés par LDA_analysis.py
TOPIC_KEYWORDS = {
    0: ["bruges", "canal", "walk", "heritage", "romantic", "history", "belfry", "medieval", "architecture", "old", "city", "stone"],
    1: ["airport", "bus", "shuttle", "station", "tour", "guide", "transport", "shuttle", "transfer", "line", "coach", "arrival"],
    2: ["museum", "art", "culture", "ticket", "pass", "exhibition", "gallery", "collection", "masterpiece", "history", "entry"],
    3: ["wine", "food", "restaurant", "local", "tasting", "gastronomy", "beer", "chocolate", "dinner", "cuisine", "specialty"],
    4: ["website", "online", "app", "information", "contact", "digital", "booking", "service", "mobile", "account", "login"],
    5: ["shopping", "theatre", "opera", "family", "kids", "attraction", "cinema", "park", "show", "performance", "store"]
}

def create_gephi_files():
    if not os.path.exists(INPUT_TFIDF):
        print(f"ERREUR : Le fichier TF-IDF est introuvable ici : {INPUT_TFIDF}")
        return

    # Charger la matrice
    df = pd.read_csv(INPUT_TFIDF, index_col=0)
    top_words = df.sum().sort_values(ascending=False).head(100).index
    df_reduced = df[top_words]

    # --- NODES ---
    nodes = []
    for word in top_words:
        assigned_topic = "Other"
        for tid, keywords in TOPIC_KEYWORDS.items():
            if any(k in word.lower() for k in keywords):
                assigned_topic = TOPIC_LABELS[tid]
                break
        nodes.append({'Id': word, 'Label': word, 'Topic': assigned_topic})
    
    pd.DataFrame(nodes).to_csv(OUTPUT_NODES, index=False, sep=';', quoting=csv.QUOTE_ALL)

    
# --- EDGES 
    co_matrix = df_reduced.T.dot(df_reduced)
    edges = []
    cols = co_matrix.columns
    
    # Option A : Baisse le seuil à 5% au lieu de 15%
    threshold = co_matrix.values.max() * 0.05 
    
    # Option B (Plus robuste) : On prend les 300 liens les plus forts
    # Cela garantit d'avoir un beau graphe dense sans être illisible
    all_edges = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            weight = co_matrix.iloc[i, j]
            if weight > 0:
                all_edges.append({'Source': cols[i], 'Target': cols[j], 'Weight': weight})
    
    # Tri et sélection des 300 meilleurs
    edges_df = pd.DataFrame(all_edges).sort_values(by='Weight', ascending=False).head(300)
    edges_df.to_csv(OUTPUT_EDGES, index=False, sep=';')


    print(f"Fichier Nodes créé : {OUTPUT_NODES}")
    print(f"Fichier Edges créé : {OUTPUT_EDGES}")

#  Evaluation metrics
import numpy as np
import networkx as nx
import pandas as pd

def degree_vector(A: np.ndarray) -> np.ndarray:
    """Vecteur des degrés ( non orienté)"""
    return A.sum(axis=1)


def transition_matrix(A: np.ndarray) -> np.ndarray:
    """
    Matrice de transition P pour PageRank sur graphe non orienté pondéré
    On normalise chaque ligne par la somme des poids sortants.
    """
    deg = A.sum(axis=1)
    P = np.zeros_like(A, dtype=float)
    for i, d in enumerate(deg):
        if d > 0:
            P[i, :] = A[i, :] / d
    return P

def pagerank_power_iteration(A: np.ndarray,
                             alpha: float = 0.85,
                             max_iter: int = 100,
                             tol: float = 1e-8) -> np.ndarray:
    """
    PageRank sur la matrice d'adjacence A.
    """
    n = A.shape[0]
    if n == 0:
        return np.array([])

    P = transition_matrix(A)
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
    Plus courts chemins (Floyd–Warshall) sur graphe non pondéré
    arête présente => dist 1
    """
    SP = (A > 0).astype(float)
    n = SP.shape[0]

    for i in range(n):
        for j in range(n):
            if SP[i, j] == 0:
                SP[i, j] = 100000.0
            else:
                SP[i, j] = 1.0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if SP[i, j] > SP[i, k] + SP[k, j]:
                    SP[i, j] = SP[i, k] + SP[k, j]

    for i in range(n):
        SP[i, i] = 0.0

    return SP


def closeness_from_shortest_paths(SP: np.ndarray) -> np.ndarray:
    """
    Closeness centrale C(i) = (n-1) / somme des distances
    """
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


def compute_betweenness(G: nx.Graph,
                        invert_weights: bool = True,
                        normalized: bool = True):
    """
    Betweenness pondérée pour graphe de similarité (poids TF-IDF)
    convertit la similarité en distance 1 - weight si invert_weights=True.
    """
    if G.number_of_nodes() == 0:
        return {}

    G_bt = G.copy()
    if invert_weights:
        for u, v, data in G_bt.edges(data=True):
            w = data.get("Weight", 1.0)
            w = min(max(w, 0.0), 1.0)
            data["weight"] = 1.0 - w

    betw = nx.betweenness_centrality(
        G_bt, weight="weight", normalized=normalized
    )
    return betw


def evaluate_tfidf_topic_graph(edges_csv: str):
    """
    Charge le fichier d'arêtes exporté (OUTPUT_EDGES) et calcule
    les métriques
    """
    df_e = pd.read_csv(edges_csv, sep=';')

    # Graphe non orienté et pondéré
    G = nx.Graph()
    for _, row in df_e.iterrows():
        G.add_edge(row["Source"], row["Target"], Weight=row["Weight"])

    results = {}

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    results["nodes"] = n_nodes
    results["edges"] = n_edges

    if n_nodes == 0:
        return results

    # Densité
    results["density"] = nx.density(G)

    # Composantes
    components = list(nx.connected_components(G))
    results["components"] = len(components)

    # Communautés + modularité
    if n_edges > 0:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities)
        results["communities"] = len(communities)
        results["modularity"] = modularity
    else:
        results["communities"] = 0
        results["modularity"] = 0.0

    # Matrice d'adjacence A
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v, data in G.edges(data=True):
        i = idx[u]
        j = idx[v]
        w = data.get("Weight", 1.0)
        A[i, j] = w
        A[j, i] = w

    # comment est le graphe globalement
    # Degré moyen
    deg_vec = degree_vector(A)
    results["avg_degree"] = float(deg_vec.mean())

    # Closeness moyenne
    SP = shortest_path_matrix(A)
    closeness_vec = closeness_from_shortest_paths(SP)
    results["avg_closeness"] = float(closeness_vec.mean())

    # Betweenness moyenne
    betw_dict = compute_betweenness(G, invert_weights=True, normalized=True)
    if betw_dict:
        results["avg_betweenness"] = float(np.mean(list(betw_dict.values())))
    else:
        results["avg_betweenness"] = 0.0

    # PageRank moyen
    pr_vec = pagerank_power_iteration(A, alpha=0.85, max_iter=200)
    results["avg_pagerank"] = float(pr_vec.mean())

    return results


if __name__ == "__main__":
    create_gephi_files()
    # Évaluation graphe 
    res = evaluate_tfidf_topic_graph(OUTPUT_EDGES)
    print("\n ÉVALUATION GRAPHE TF-IDF / LDA ")
    for k, v in res.items():
        print(f"{k} : {v}")

