"""
Builds a lexical co-occurrence network using a sliding window over tokens and exports it to Gephi-compatible CSVs.

The script:
- loads and cleans the corpus from a JSON file (one document per page),
- keeps only the top-N most frequent tokens,
- builds a symmetric sparse co-occurrence matrix with a sliding window,
- converts it to a similarity matrix using either Jaccard or Cosine,
- constructs an undirected NetworkX graph, evaluates basic network metrics,
- shows a preview of communities, and exports nodes/edges CSVs for Gephi.

Inputs
- json_path :
    Path to the cleaned corpus JSON, e.g. "data/processed/corpus_json.json".
    The JSON is loaded and preprocessed by load_and_clean_json from clean_tokens_cooc.
- Parameters:
    window_size (int): size of the sliding window for co-occurrence.
    top_N (int): number of most frequent tokens to keep.
    suffix (str): optional suffix to differentiate runs (e.g. "global", "sea").
    Interactive choice between:
        1 = Jaccard similarity on binarized co-occurrence.
        2 = Cosine similarity on the co-occurrence matrix.

Outputs
- CSV files (Gephi-ready) written to:
    data/link_analysis/coocurence_window_graph/nodes_{metric_name}_{suffix}.csv
    data/link_analysis/coocurence_window_graph/edges_{metric_name}_{suffix}.csv
  where metric_name ∈ {"jaccard", "cosine"} and suffix is optional.
- Console output:
    Basic network metrics (nodes, edges, density, modularity, etc.)
    Text preview of top terms per community by degree.
- Return value:
    The constructed NetworkX graph G.
"""
import json
import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities
from networkx.convert_matrix import from_scipy_sparse_array


# clean script
from clean_tokens_cooc import load_and_clean_json


# PART 1: MATRIX CONSTRUCTION (Sliding Window)
def build_sparse_cooccurrence(token_sequences, window_size=7):
    """
    Creates a sparse co-occurrence matrix using a sliding window.
    """
    print(f"Building Co-occurrence Matrix (Window Size: {window_size})...")
    
    
    terms = sorted(set(t for doc in token_sequences for t in doc))
    term2idx = {t: i for i, t in enumerate(terms)}
    N = len(terms)

    
    cooc = dok_matrix((N, N), dtype=np.float32)

    # Sliding window logic
    for tokens in token_sequences:
        for i, w1 in enumerate(tokens):
            if w1 not in term2idx: continue 
            w1_idx = term2idx[w1]
            
            
            window = tokens[i+1 : i+1+window_size]
            for w2 in window:
                if w2 not in term2idx: continue
                if w1 == w2: continue 
                
                w2_idx = term2idx[w2]
                cooc[w1_idx, w2_idx] += 1
                cooc[w2_idx, w1_idx] += 1  

    # Convert to CSR for fast calculations
    return cooc.tocsr(), terms

# PART 2: Similarity measure (Jaccard & Cosine)

def jaccard_sparse(cooc_csr, min_jaccard=0.1):
    """Optimized Jaccard calculation for Sparse Matrices"""
    print("Calculating Jaccard Similarity...")
    
    # Binarize (Presence/Absence)
    binary = (cooc_csr > 0).astype(np.int8)
    intersect = binary.T @ binary
    degrees = np.array(binary.sum(axis=0)).flatten()
    
    intersect_coo = intersect.tocoo()
    r, c, v = intersect_coo.row, intersect_coo.col, intersect_coo.data
    
    union = degrees[r] + degrees[c] - v
    
   
    scores = v / union
    
    
    mask = scores >= min_jaccard
    
    return csr_matrix((scores[mask], (r[mask], c[mask])), shape=intersect.shape)

def cosine_sparse_calculation(cooc_csr, min_cosine=0.25):
    """Optimized Cosine Similarity calculation"""
    print("Calculating Cosine Similarity...")
    
    
    sim_matrix = cosine_similarity(cooc_csr, dense_output=False)
    
    
    print(f"Filtering Cosine results (strict threshold > {min_cosine})...")
    
    # In-place operation for efficiency
    sim_matrix.data[sim_matrix.data < min_cosine] = 0
    sim_matrix.eliminate_zeros()
    
    return sim_matrix

# PART 3: GRAPH CONSTRUCTION & EVALUATION

def build_graph_from_matrix(sim_matrix, terms, min_degree=3):
    """Generic function: works for both Jaccard and Cosine"""

    print("Converting to NetworkX Graph...")
    G = from_scipy_sparse_array(sim_matrix)

    mapping = {i: t for i, t in enumerate(terms)}
    G = nx.relabel_nodes(G, mapping)

    # Cleanup
    print("Removing isolated and weak nodes...")
    G.remove_nodes_from(list(nx.isolates(G)))
    G.remove_nodes_from([n for n, d in G.degree() if d < min_degree])

    return G

def evaluate_network(G):
    print("Calculating network metrics...")
    density = nx.density(G)
    components = list(nx.connected_components(G))
    
    modularity = 0
    if G.number_of_edges() > 0:
        communities = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities)
    else:
        communities = []
        
    
    nb_nodes = G.number_of_nodes()

    if nb_nodes > 0:
        largest_cc_size = len(max(nx.connected_components(G), key=len))
        largest_component_ratio = largest_cc_size / nb_nodes
    else:
        largest_component_ratio = 0

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": density,
        "modularity": modularity,
        "components": len(components),
        "communities_count": len(communities),
        "largest_component_ratio": largest_component_ratio
    }


def top_n_terms(token_sequences, N=10000):
    from collections import Counter
    counter = Counter(t for doc in token_sequences for t in doc)
    most_common_terms = set([t for t, _ in counter.most_common(N)])
    
    # Filter sequences
    filtered_sequences = [[t for t in doc if t in most_common_terms] for doc in token_sequences]
    return filtered_sequences, list(most_common_terms)

def show_communities_by_centrality(G):
    if G.number_of_nodes() == 0:
        return
    communities = list(greedy_modularity_communities(G))
    node_degrees = dict(G.degree())
    
    # Show top 10 communities
    for i, comm in enumerate(communities, start=1): 
        sorted_terms = sorted(list(comm), key=lambda x: node_degrees[x], reverse=True)
        print(f"\n--- Community {i} ({len(comm)} terms) ---")
        print(", ".join(sorted_terms[:20]))

# PART 4: GEPHI EXPORT
def export_to_gephi_csv(
    G,
    output_folder="data/link_analysis/coocurence_window_graph",
    metric_name="similarity",
    suffix="",
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if suffix:
        print(f"Exporting to Gephi format in '{output_folder}' with suffix '_{metric_name}_{suffix}'...")
    else:
        print(f"Exporting to Gephi format in '{output_folder}' with suffix '_{metric_name}'...")

    if G.number_of_nodes() == 0:
        print("Graph is empty, skipping export.")
        return

    communities = list(greedy_modularity_communities(G))
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    # Export Nodes
    nodes_data = []
    weighted_degrees = dict(G.degree(weight='weight'))
    for node in G.nodes():
        nodes_data.append({
            "Id": node,
            "Label": node,
            "Modularity Class": node_community.get(node, 0),
            "Degree": G.degree(node),
            "Weighted Degree": weighted_degrees.get(node, 0),
        })

    if suffix:
        nodes_filename = f"{output_folder}/nodes_{metric_name}_{suffix}.csv"
    else:
        nodes_filename = f"{output_folder}/nodes_{metric_name}.csv"
    pd.DataFrame(nodes_data).to_csv(nodes_filename, index=False)
    
    # Export Edges
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "Source": u,
            "Target": v,
            "Type": "Undirected",
            "Weight": data.get('weight', 1.0),
        })

    if suffix:
        edges_filename = f"{output_folder}/edges_{metric_name}_{suffix}.csv"
    else:
        edges_filename = f"{output_folder}/edges_{metric_name}.csv"
    pd.DataFrame(edges_data).to_csv(edges_filename, index=False)

    print(f"Export complete: {nodes_filename}, {edges_filename}")


# PART 5: INTERACTIVE PIPELINE
def run_pipeline_interactive(json_path, window_size=7, top_N=10000, suffix=""):
    # 1. Loading
    print("1. Loading Corpus...")
    token_sequences = load_and_clean_json(json_path)

    # 2. Filtering Top N
    print(f"2. Keeping top {top_N} most frequent words...")
    token_sequences, terms = top_n_terms(token_sequences, N=top_N)

    # 3. Co-occurrence
    cooc_csr, terms = build_sparse_cooccurrence(token_sequences, window_size=window_size)

    # 4. Metric Selection
    print("\n" + "="*50)
    print("CHOOSE YOUR METRIC:")
    print("  Type '1' for JACCARD (Strict Intersection)")
    print("  Type '2' for COSINE (Vector Similarity)")
    print("="*50)

    choice = input("Enter choice (1 or 2): ").strip()

    sim_matrix = None
    min_degree = 5

    if choice == "1":
        print("\n--> Mode JACCARD selected.")
        sim_matrix = jaccard_sparse(cooc_csr, min_jaccard=0.15)
        metric_name = "jaccard"

    elif choice == "2":
        print("\n--> Mode COSINE selected.")
        sim_matrix = cosine_sparse_calculation(cooc_csr, min_cosine=0.35)
        metric_name = "cosine"

    else:
        print("Invalid choice. Please restart.")
        return

    # 5. Graph Construction
    G = build_graph_from_matrix(sim_matrix, terms, min_degree=min_degree)

    # 6. Evaluation
    print("\n--- NETWORK EVALUATION ---")
    results = evaluate_network(G)
    for k, v in results.items():
        print(f"{k} : {v}")

    # 7. Community Preview
    print("\n--- CLUSTER PREVIEW ---")
    show_communities_by_centrality(G)

    # 8. Export
    export_to_gephi_csv(G, metric_name=metric_name, suffix=suffix)

    return G

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    G = run_pipeline_interactive(
        json_path="data/processed/corpus_json.json",
        window_size=7,
        top_N=10000
    )