import pandas as pd
import numpy as np
import networkx as nx
import os
import time


# 1. DATA LOADING (PANDAS -> NUMPY/NETWORKX)

# Adjacency Matrix construction from Gephi CSV exports
def load_gephi_data(nodes_path, edges_path, directed=False):
    print(f"Loading {nodes_path} and {edges_path}...")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"ERROR: Files not found. Please check the 'gephi_export' folder.")
        return None, None, None

    # Loading
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Cleaning IDs 
    nodes_df['Id'] = nodes_df['Id'].astype(str)
    edges_df['Source'] = edges_df['Source'].astype(str)
    edges_df['Target'] = edges_df['Target'].astype(str)

    # Mapping: Gephi ID -> Integer Index (0, 1, 2...)
    id_to_idx = {id_val: i for i, id_val in enumerate(nodes_df['Id'])}
    
    # Inverse Mapping for display: Index -> Label (Keyword)
    if 'Label' in nodes_df.columns:
        idx_to_label = nodes_df['Label'].fillna(nodes_df['Id']).tolist()
    else:
        idx_to_label = nodes_df['Id'].tolist()

    n = len(nodes_df)

    # --- Construction of Matrix A (Numpy) ---
    A = np.zeros((n, n), dtype=float)

    for _, row in edges_df.iterrows():
        source = row['Source']
        target = row['Target']
        
        if source in id_to_idx and target in id_to_idx:
            u = id_to_idx[source]
            v = id_to_idx[target]
            weight = float(row.get('Weight', 1.0))
            
            A[u, v] = weight
            if not directed:
                A[v, u] = weight

    # --- Construction of Graph G (NetworkX) ---
    id_to_label_map = dict(zip(nodes_df['Id'], nodes_df['Label'] if 'Label' in nodes_df else nodes_df['Id']))
    
    edges_with_labels = []
    for _, row in edges_df.iterrows():
        s = id_to_label_map.get(row['Source'])
        t = id_to_label_map.get(row['Target'])
        w = float(row.get('Weight', 1.0))
        if s and t:
            edges_with_labels.append((s, t, w))

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
        
    G.add_weighted_edges_from(edges_with_labels)

    return A, G, idx_to_label


# 2. CENTRALITY MEASURES (NUMPY)

def degree_matrix(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=1)

def transition_matrix(A: np.ndarray) -> np.ndarray:
    degrees = degree_matrix(A)
    degrees[degrees == 0] = 1 
    P = A / degrees[:, np.newaxis]
    return P

def pagerank_power_iteration(A: np.ndarray, alpha: float = 0.85, max_iter: int = 100) -> np.ndarray:
    P = transition_matrix(A)
    n = A.shape[0]
    pr = np.ones((n, 1)) / n
    E = np.ones((n, n)) / n
    G_matrix = alpha * P + (1 - alpha) * E
    
    for _ in range(max_iter):
        pr = G_matrix.T @ pr
        pr = pr / pr.sum()
            
    return pr.flatten()


def compute_betweenness(G):
    nb_nodes = G.number_of_nodes()
    
    print(f"\n   [Betweenness] Starting on {nb_nodes} nodes.")
    if nb_nodes == 0: return {}
    
    # 1. Inversion of weights (Similarity -> Distance)
    G_dist = G.copy()
    for u, v, data in G_dist.edges(data=True):
        w = data.get("weight", 0.01)
        if w > 0:
            data["weight"] = 1.0 / w 
        else:
            data["weight"] = 100.0

    # 2. Calculation with Approximation (time saving)
    k_val = 1500 if nb_nodes > 2000 else None
    if k_val:
        print(f"Large graph. Using approximation k={k_val}.)")

    t0 = time.time()
    betweenness = nx.betweenness_centrality(G_dist, weight="weight", normalized=True, k=k_val, seed=42)
    print(f"   [Betweenness] Finished in {time.time() - t0:.2f} seconds.")

    return betweenness

def shortest_path(G, target_node, comparison_nodes=None):
    target_node_real = None
    for n in G.nodes():
        if n.lower() == target_node.lower():
            target_node_real = n
            break
            
    if not target_node_real:
        print(f"   [!] Node '{target_node}' not found in this graph.")
        return

    # Creation of distance graph
    dist_graph = nx.Graph()
    for u, v, data in G.edges(data=True):
        sim = data.get('weight', 0.0)
        dist = 1.0 - sim + 0.0001 
        dist_graph.add_edge(u, v, weight=dist)

    print(f"\n   --- Positioning Analysis for: {target_node_real} ---")
    
    try:
        lengths = nx.shortest_path_length(dist_graph, source=target_node_real, weight='weight')
        
        if comparison_nodes:
            results = []
            for city in comparison_nodes:
                real_city = None
                for n in G.nodes():
                    if n.lower() == city.lower():
                        real_city = n
                        break
                
                if real_city and real_city in lengths:
                    d = lengths[real_city]
                    path = nx.shortest_path(dist_graph, source=target_node_real, target=real_city, weight='weight')
                    concepts = path[1:-1] if len(path) > 2 else ["(Direct Link)"]
                    results.append((city, d, concepts))
                else:
                    results.append((city, 999.0, ["Not Connected"]))
            
            results.sort(key=lambda x: x[1])
            for city, d, concepts in results:
                concept_str = ", ".join(concepts[:3])
                dist_str = f"{d:.2f}" if d < 900 else "INF"
                print(f"   > {city.ljust(12)} : Dist {dist_str} | Via: {concept_str}")
                
    except Exception as e:
        print(f"Error calculating path: {e}")

# esthetic printing
def print_top_nodes(scores, labels, title, n=10):
    print(f"\n   --- {title} (Top {n}) ---")
    if isinstance(scores, dict):
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for label, val in sorted_items[:n]:
            print(f"   {label} : {val:.4f}")
    elif isinstance(scores, (list, np.ndarray)):
        paired = list(zip(labels, scores))
        sorted_items = sorted(paired, key=lambda x: x[1], reverse=True)
        for label, val in sorted_items[:n]:
            print(f"   {label} : {val:.4f}")


# 4. MAIN
def run_evaluation():
    print("========================================================")
    print("      Jaccard evaluation           ")
    print("========================================================")

    concurrents = ["Barcelona", "Lisbon", "Rome", "Amsterdam", "Valencia", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"]
    
    
    A_jac, G_jac, labels_jac = load_gephi_data("gephi_export/nodes_jaccard.csv", "gephi_export/edges_jaccard.csv")

    if A_jac is not None:
        # A. Degree
        print_top_nodes(degree_matrix(A_jac), labels_jac, "Degree Centrality")
        # B. PageRank
        print_top_nodes(pagerank_power_iteration(A_jac), labels_jac, "PageRank")
        
        # C. Betweenness & Shortest Path 
        bt_jac = compute_betweenness(G_jac)
        print_top_nodes(bt_jac, labels_jac if isinstance(labels_jac, list) else list(G_jac.nodes()), "Betweenness Jaccard")
        
        
        shortest_path(G_jac, "bruges", concurrents)
        
    
    print("\n" + "="*50)
    print(" Cosine evaluation")
    print("="*50)
    A_cos, G_cos, labels_cos = load_gephi_data("gephi_export/nodes_cosine.csv", "gephi_export/edges_cosine.csv")

    if A_cos is not None:
        # A. degree and pagerank
        print_top_nodes(degree_matrix(A_cos), labels_cos, "Degree Cosine")
        print_top_nodes(pagerank_power_iteration(A_cos), labels_cos, "PageRank Cosine")

        # B. shortest path
        shortest_path(G_cos, "bruges", concurrents)

        # C. betweenness
        bt = compute_betweenness(G_cos)
        print_top_nodes(bt, labels_cos if isinstance(labels_cos, list) else list(G_cos.nodes()), "Betweenness Cosine")

if __name__ == "__main__":
    run_evaluation()


