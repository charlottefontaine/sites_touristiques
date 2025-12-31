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

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": density,
        "modularity": modularity,
        "components": len(components),
        "communities_count": len(communities)
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
    for i, comm in enumerate(communities[:10], start=1): 
        sorted_terms = sorted(list(comm), key=lambda x: node_degrees[x], reverse=True)
        print(f"\n--- Community {i} ({len(comm)} terms) ---")
        print(", ".join(sorted_terms[:20]))

# PART 4: GEPHI EXPORT
def export_to_gephi_csv(G, output_folder="gephi_export"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Exporting to Gephi format in '{output_folder}'...")
    
    if G.number_of_nodes() == 0:
        print("Graph is empty, skipping export.")
        return

    communities = list(greedy_modularity_communities(G))
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

   
    nodes_data = []
    weighted_degrees = dict(G.degree(weight='weight'))
    for node in G.nodes():
        nodes_data.append({
            "Id": node, "Label": node,
            "Modularity Class": node_community.get(node, 0),
            "Degree": G.degree(node),
            "Weighted Degree": weighted_degrees.get(node, 0)
        })
    pd.DataFrame(nodes_data).to_csv(f"{output_folder}/nodes.csv", index=False)

    
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "Source": u, "Target": v, "Type": "Undirected",
            "Weight": data.get('weight', 1.0)
        })
    pd.DataFrame(edges_data).to_csv(f"{output_folder}/edges.csv", index=False)
    print("Export complete.")

# PART 5: INTERACTIVE PIPELINE
def run_pipeline_interactive(json_path, window_size=7, top_N=10000):
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
        sim_matrix = jaccard_sparse(cooc_csr, min_jaccard=0.08) 
        
    elif choice == "2":
        print("\n--> Mode COSINE selected.")
        sim_matrix = cosine_sparse_calculation(cooc_csr, min_cosine=0.35)
        
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
    export_to_gephi_csv(G, "gephi_export")

    return G

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    G = run_pipeline_interactive(
        json_path="data/processed/corpus_json.json",
        window_size=7,
        top_N=10000
    )

