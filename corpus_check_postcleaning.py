import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
BASE_PATH = "data/processed"
TDM_PATH = os.path.join(BASE_PATH, "df_freq_terms.csv")  # Input: Cleaned TDM
OUTPUT_DIR = "data/diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# --------------------------------------------------------------------------

def profile_matrix(tdm: pd.DataFrame):
    """
    Displays fundamental statistics about the term-document matrix.
    
    Input:
        tdm: pandas DataFrame (rows=documents, cols=terms)
    Output:
        prints statistics to console
    """
    print("1) Matrix Profile ")
    
    n_docs, n_terms = tdm.shape
    total_cells = n_docs * n_terms
    non_zero_cells = tdm.astype(bool).sum().sum()
    
    # Sparsity = percentage of zero elements
    sparsity = 1.0 - (non_zero_cells / total_cells)
    
    print(f"Number of documents: {n_docs}")
    print(f"Vocabulary size (unique terms): {n_terms}")
    print(f"Matrix sparsity: {sparsity:.2%}")


def analyze_document_lengths(tdm: pd.DataFrame):
    """
    Analyzes and visualizes the distribution of document lengths (word count).
    
    Input:
        tdm: pandas DataFrame
    Output:
        displays plot and prints stats
    """
    print("2. Document Length Analysis")
    
    # Calculate word count per document (sum of row)
    doc_lengths = tdm.sum(axis=1)
    
    print(f"Average words per document: {doc_lengths.mean():.2f}")
    print(f"Shortest document (words): {doc_lengths.min()}")
    print(f"Longest document (words): {doc_lengths.max()}")
    
    # Identify potentially problematic short documents
    short_docs = doc_lengths[doc_lengths < 20]
    if not short_docs.empty:
        print(f"\n {len(short_docs)} documents have fewer than 20 words.")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(doc_lengths, bins=50, kde=True)
    plt.title("Distribution of Document Lengths (Word Count)")
    plt.xlabel("Number of words per document")
    plt.ylabel("Number of documents")
    
    save_path = os.path.join(OUTPUT_DIR, "doc_length_distribution.png")
    plt.savefig(save_path)
    plt.show()

def analyze_term_frequency(tdm: pd.DataFrame):
    """
    Analyzes term frequency distribution to check Zipf's Law.
    
    Input:
        tdm: pandas DataFrame
    Output:
        displays log-log plot
    """
    print("3. Term Frequency Analysis (Zipf's Law)")
    
    # Sum frequencies for each term and sort descending
    term_frequencies = tdm.sum().sort_values(ascending=False)
    
    print(f"Most frequent term: '{term_frequencies.index[0]}' ({term_frequencies.iloc[0]} times)")
    print(f"Least frequent term: '{term_frequencies.index[-1]}' ({term_frequencies.iloc[-1]} times)")
    
    # Visualization (Log-Log scale)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(term_frequencies)), term_frequencies.values)
    
    plt.title("Term Frequency Distribution (Zipf's Law)")
    plt.xlabel("Term Rank (Most to Least Frequent)")
    plt.ylabel("Total Frequency (Log Scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    
    save_path = os.path.join(OUTPUT_DIR, "zipf_distribution.png")
    plt.savefig(save_path)
    plt.show()
    
def analyze_term_lengths(tdm: pd.DataFrame):
    """
    Analyzes the distribution of term lengths (number of characters)
    To check if short words/stopwords were correctly removed
    
    Input:
        tdm: pandas DataFrame
    Output:
        displays histogram
    """
    print("4. Term Length Analysis")

    term_lengths = pd.Series([len(term) for term in tdm.columns])

    print(f"Average term length: {term_lengths.mean():.2f} chars")
    print(f"Min length: {term_lengths.min()} chars")
    print(f"Max length: {term_lengths.max()} chars")

    short_terms = [term for term in tdm.columns if len(term) < 3]
    print(f"Terms with < 3 characters: {len(short_terms)}")

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(term_lengths, bins=20, discrete=True)
    plt.title("Distribution of Term Lengths (Characters)")
    plt.xlabel("Number of characters per term")
    plt.ylabel("Number of terms")
    
    save_path = os.path.join(OUTPUT_DIR, "term_length_distribution.png")
    plt.savefig(save_path)
    plt.show()


#  EXECUTION
# --------------------------------------------------------------------------

def main():
    if not os.path.exists(TDM_PATH):
        print(f"Error: File '{TDM_PATH}' not found.")
        print("Please run 'corpus_cleaning.py' first.")
        return

    print(f"Loading data from {TDM_PATH}...")
    tdm = pd.read_csv(TDM_PATH, index_col=0)
    print("Data loaded successfully.\n")

    # Run diagnostics
    profile_matrix(tdm)
    analyze_document_lengths(tdm)
    analyze_term_frequency(tdm)
    analyze_term_lengths(tdm)

if __name__ == "__main__":
    main()
