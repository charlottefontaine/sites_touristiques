import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.pairwise import cosine_similarity


def load_tdm(csv_path):
    """
    Load a cleaned term-document matrix (TDM)
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df


'''
def build_cooccurrence_from_tdm(df):
    """
    Build a term-term co-occurrence matrix
    from a term-document matrix (documents x terms)
    """
    # binary presence / absence
    binary = (df > 0).astype(int)

    # co-occurrence = TDMᵀ × TDM
    cooc = binary.T @ binary

    # remove diagonal
    np.fill_diagonal(cooc.values, 0)

    cooc.to_csv("data/processed/cooccurrence_matrix.csv")
    return cooc
'''


def build_cooccurrence_from_tdm(df, min_cooc=1):
    """
    Term-term co-occurrence matrix with a minimum co-occurrence threshold
    """
    binary = (df > 0).astype(int)
    cooc = binary.T @ binary

    # remove diagonal
    np.fill_diagonal(cooc.values, 0)

    # apply minimum co-occurrence threshold
    cooc[cooc < min_cooc] = 0

    return cooc


def jaccard_from_cooccurrence(cooc_df):
    """
    Compute Jaccard similarity between terms
    """
    cooc = cooc_df.values
    term_freq = np.diag(cooc_df.values + np.eye(cooc.shape[0]))

    intersection = cooc
    union = term_freq[:, None] + term_freq[None, :] - intersection

    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0
    )

    return pd.DataFrame(
        jaccard,
        index=cooc_df.index,
        columns=cooc_df.columns
    )


def build_graph_from_jaccard(jaccard_df, min_jaccard=0.05, min_degree=3):
    """
    Convert a Jaccard similarity matrix into a NetworkX graph
    """
    G = nx.Graph()

    terms = jaccard_df.index.tolist()
    G.add_nodes_from(terms)

    for i, t1 in enumerate(terms):
        for j, t2 in enumerate(terms):
            if i < j and jaccard_df.iat[i, j] >= min_jaccard:
                G.add_edge(t1, t2, weight=jaccard_df.iat[i, j])

    # remove low-degree nodes
    G.remove_nodes_from([n for n, d in G.degree() if d < min_degree])

    return G


def evaluate_network(G):
    """
    Evaluate the network: density, modularity, connected components
    """
    density = nx.density(G)

    components = list(nx.connected_components(G))
    n_components = len(components)

    if G.number_of_edges() > 0:
        communities = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities)
    else:
        modularity = 0.0
        communities = []

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": density,
        "components": n_components,
        "communities": len(communities),
        "modularity": modularity
    }


def run_cooccurrence_pipeline(csv_path,
                              min_jaccard=0.15,
                              min_degree=3):

    print("Loading TDM...")
    df = load_tdm(csv_path)

    print("Building co-occurrence matrix...")
    cooc = build_cooccurrence_from_tdm(df, min_cooc=2)

    print("Computing Jaccard similarity...")
    jaccard = jaccard_from_cooccurrence(cooc)

    print("Building graph...")
    G = build_graph_from_jaccard(
        jaccard,
        min_jaccard=min_jaccard,
        min_degree=min_degree
    )

    print("Evaluating network...")
    results = evaluate_network(G)

    return G, results


G, results = run_cooccurrence_pipeline(
    csv_path="data/processed/df_freq_terms.csv",
    min_jaccard=0.1,
    min_degree=3,
)

print("\n--- NETWORK EVALUATION ---")
for k, v in results.items():
    print(f"{k} : {v}")
