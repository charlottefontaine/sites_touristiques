<p align="center">
<img width="233" height="179" alt="Logo ACEClick" src="https://github.com/user-attachments/assets/65e289c5-6f2e-4c91-9e2f-f6556891802d" />

</p>

<h1 align="center">Web Mining: Assessing the Online Attractiveness of European Tourism Sites</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Status-Project_Completed-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/Analysis-Web_Scraping_%7C_Text_Mining_%7C_Link_Analysis-orange.svg" alt="Analysis Type">
  <img src="https://img.shields.io/badge/University-UCLouvain-maroon.svg" alt="Louvain Scool Of Management">
</p>

<p align="center">
  <b>A comparative analysis of digital strategies to enhance the online positioning of Bruges.</b>
</p>

# Contents Overview
- [🚀 Project Overview](#-project-overview)
- [🛠 Methodology & Pipeline](#-methodology--pipeline)
- [📂 Repository Structure](#-repository-structure)
- [📊 Key Visualizations](#-key-visualizations)
- [👥 ACE Click Team](#-ace-click-team)

***

# 🚀 Project Overview
##### [:rocket: Go to Contents Overview](#contents-overview)

In a European market welcoming over 340 million international tourists, official websites have become strategic digital storefronts. This project, led by the **ACE Click** consultancy, benchmarks **10 major European destinations** to identify digital best practices.

**The Challenge:** How do European tourist sites structure and promote their content, and what levers can **Bruges** use to strengthen its digital appeal?

**Cities Analyzed:** Barcelona, Lisbon, Rome, Ostend, Amsterdam, Valencia, Copenhagen, Manchester, Cologne, and **Bruges**.

***
# ⚡ Quick Start
##### [:rocket: Go to Contents Overview](#contents-overview)

This quick start reproduces the full analytical pipeline, from web scraping to semantic and lexical network analysis.
## Requirements
Python 3.9+

## Install dependencies

```bash
pip install pandas numpy matplotlib seaborn networkx scipy scikit-learn nltk requests beautifulsoup4 openpyxl wordcloud
```

## Configure cities, websites and categories
parameters.py

## Scrape tourism websites (depth = 1)
python run_scrapping.py

## Build, clean and validate the corpus
python corpus_analysis.py
python corpus_cleaning.py
python corpus_check_post_cleaning.py
## Text Mining, Semantic & Network Analyses
Once the corpus is cleaned and validated, run the text mining and semantic analysis modules to extract thematic structures (TF-IDF, LDA, N-grams, concordances) and perform clustering and similarity analyses between cities.
For lexical network analysis, tokens must first be filtered and lemmatized using clean_tokens_cooc.py before running the co-occurrence pipeline (coocurence_window.py and related scripts), which generates Jaccard and Cosine similarity graphs exported for Gephi visualization.

***
# 🛠 Methodology & Pipeline
##### [:rocket: Go to Contents Overview](#contents-overview)

The project follows a rigorous data pipeline, leveraging Python's most powerful libraries for data science, machine learning, and graph theory.

### 1. Web Scraping & Data Engineering
- **Strategy**: Automated "Wide-but-shallow" crawling (Depth 1) to target the most strategic content curated by tourism boards.
- **Data Structuring**: 894 documents processed and organized using `Pandas`, `NumPy`, and `JSON` for structured storage. 
- **Efficiency**: Management of high-volume extractions with `time` and `os` modules.
- **Tools**: `BeautifulSoup`, `Requests`, `Selenium`, `Pandas`, `NumPy`, `JSON`.


### 2. Text Mining & Machine Learning
- **Preprocessing**: Advanced NLP pipeline using `NLTK`, `re`, and custom `Stopwords`. This includes multilingual noise filtering and advanced **Lemmatization** (benchmarked against Stemming).
- **Statistical Quality Control**: Validation of the corpus using frequency distributions (`Counter`, `defaultdict`) and Zipf’s Law diagnostics.
- **Advanced Modeling**:
  - **Vectorization**: **TF-IDF** calculated from `csr_matrix` and `dok_matrix` (SciPy) for high-dimensional efficiency.
  - **Thematic Modeling**: **Latent Dirichlet Allocation (LDA)** to discover latent editorial pillars.
  - **Clustering & Dim. Reduction**: City profile segmentation using **K-Means clustering** and **PCA** (Principal Component Analysis) to visualize cluster variance.
- **Tools**: `Scikit-learn`, `SciPy`, `NLTK`, `Seaborn`, `Matplotlib`.


### 3. Link & Network Analysis
- **Structural Mapping**: Modeling of site architecture and page hierarchy using `NetworkX` to analyze source-target relationships.
- **Network Metrics**: 
  - **Centrality**: Calculation of connectivity scores and **PageRank** to identify content hubs.
  - **Community Detection**: Implementation of `greedy_modularity_communities` to find thematic clusters within the link structure.
- **Advanced Graphs**: Exporting co-occurrence data and similarity scores (`cosine_similarity`) for high-fidelity visualization in **Gephi**.
- **Tools**: `NetworkX`, `SciPy (sparse matrices)`, `Scikit-learn`, `Gephi`.

# 📂 Repository Structure
##### [:rocket: Go to Contents Overview](#contents-overview)
### Data folders
- **`data/raw/`**  
  Contains all *unprocessed* inputs scraped from tourism websites.
  - **`data/raw/html/homepages/`**  
    Homepage HTML files for each city.
  - **`data/raw/html/depth1/`**  
    Depth‑1 HTML files (internal pages linked from the homepage).
- **`data/processed/`**  
  Central repository of *cleaned* and *structured* data used as inputs for multiple analyses.
  - **`data/processed/corpus.csv`**  
    Flat text corpus (one row per page) with columns such as `city`, `url`, `depth`, `text`, `word_count`.
  - **`data/processed/corpus_json.json`**  
    JSON version of the corpus, one object per page, used for flexible text processing.
  - **`data/processed/corpus_sea.json`, `corpus_nosea.json`, `corpus_north.json`, `corpus_south.json`**  
    Sub‑corpora for the four geographic / coastal categories (Sea, No‑Sea, North, South).
  - **`data/processed/df_freq_terms.csv`**  
    Document–term frequency matrix (rows = pages, columns = terms, plus a `city` column).
  - **`data/processed/tfidf_by_city_norm.csv`**  
    City‑level normalized TF‑IDF matrix (rows = cities, columns = terms).
- **`data/diagnostic/`**   
  Optional folder for intermediate diagnostic exports (e.g. sparsity reports, Zipf‑law plots, matrix summaries) produced during quality checks.
- **`data/text_analysis/`**  
  Outputs from text mining and semantic analyses (TF‑IDF, LDA, N‑grams, clustering, word clouds, concordances, sentiment).
  - **`data/text_analysis/concordances/`**  
    KWIC tables for strategic keywords (e.g. `concordance_romantic.csv`, `concordance_shopping.csv`).
  - **`data/text_analysis/classification/`**  
    Topic scores per city and dominant topic labels (e.g. `city_topic_scores.csv`, `city_topic_scores.xlsx`).
  - **`data/text_analysis/lda/`**  
    LDA‑related exports such as topic–word lists and topic–city heatmaps.
  - **`data/text_analysis/ngrams/`**  
    Filtered N‑gram (bigram) tables by theme.
  - **`data/text_analysis/clustering/`**  
    Dendrogram and similarity heatmap figures for hierarchical clustering of cities.
  - **`data/text_analysis/wordclouds/`**  
    Word cloud images for cities and geographic categories.
  - **`data/text_analysis/tfidf_top_terms/`**  
    CSVs and figures with top TF‑IDF terms by city or category.
- **`data/link_analysis/`**  
  Outputs related to co‑occurrence graphs and lexical network analysis (web mining part).
  - **`data/link_analysis/coocurence_window_graph/`**  
    Node and edge CSVs for the global and category‑specific co‑occurrence graphs (Jaccard / Cosine), plus derived matrices.
    - `nodes_jaccard.csv`, `edges_jaccard.csv`  
      Global lexical co‑occurrence graph (Jaccard).
    - `nodes_cosine.csv`, `edges_cosine.csv`  
      Global lexical co‑occurrence graph (Cosine).
    - `nodes_jaccard_sea.csv`, `edges_jaccard_sea.csv`  
      Co‑occurrence graph restricted to Sea cities (and analogous files for No‑Sea, North, South).
    - `nodes_jaccard_brugeShare.csv`  
      Jaccard node file enriched with the `bruges_share` score per term.
    - `city_community_matrix_jaccard.csv`  
      City–community weight matrix derived from the global Jaccard graph.
### Codes
#### 1. Tools
* **`parameters.py`**: Maps cities to their tourism websites and categorizes them by geography and coastal access.
* **`Utils.py`**: Tools to scrape tourism websites and preprocess the text for NLP analysis.
#### 2. Scraping
* **`run_scrapping.py`**: Automated harvesting script extracting editorial content from the 10 target cities' tourism websites into a CSV corpus.
#### 3. Corpus treatment
* **`corpus_analysis.py`**: Converts the corpus to JSON and generates an Excel report containing detailed statistical metrics on word counts and page distributions per city.
* **`corpus_cleaning.py`**: Cleans the corpus (tokenization, stopwords, lemmatization), builds the document–term frequency matrix `df_freq_terms.csv` and the normalized TF‑IDF matrix by city,
* generates TF-IDF visualizations to identify top keywords across city categories.
* **`corpus_check_post_cleaning.py`**: Performs quality control and statistical diagnostics on the cleaned matrix, including sparsity checks and Zipf's Law validation.
* **`compare_lemmatization_stemming.py`**: Compares lemmatization and stemming methods to evaluate their impact on vocabulary size and matrix sparsity.
* **`categories_cleaning.py`**: Computes TF-IDF scores and splits the term-document matrix into geographical subsets (North, South, Sea, and No-Sea) for comparative analysis.
#### 4. Text Mining
* **`word_clouds.py`**: Generates visual word clouds for each city category to highlight the most frequent terms across geographical zones.
* **`LDA_analysis.py`**: Performs Latent Dirichlet Allocation to discover thematic topics across the corpus and visualizes their distribution per city using heatmaps.
*  **`Classification.py`**: Scoring and normalizing TF-IDF topics to confirm LDA results.
* **`Hyperparams_optimization_hierarchical_clust.py`**: Executes hierarchical clustering and generates dendrograms to visualize city similarities based on diverse distance metrics and linkage methods.
* **`hierarchical clustering and similarity analysis.py`**: Maps city similarities using cosine heatmaps and hierarchical dendrograms based on TF-IDF profiles.
* **`N-grams_analysis.py`**: Extracts and filters word pairs (bigrams) associated with specific tourism themes like "shopping" or "romantic" to identify local strategic trends.
* **`concordance_analysis.py`**: Generates Keyword-in-Context (KWIC) tables to analyze how specific terms like "romantic" or "shopping" are used across different city subsets.
* **`sentiment lexicon analysis.py`**:Comparative sentiment analysis pipeline using VADER and custom emotional lexicons.
#### 5. Link analysis
* **`coocurence_window.py`**: Builds a sparse co-occurrence matrix using a sliding window to generate Jaccard or Cosine similarity graphs and identifies semantic clusters.
* **`clean_tokens_cooc.py`**: Filters out CSS/HTML technical artifacts, multilingual noise, and generic tourism stopwords while lemmatizing English tokens to prepare a high-quality corpus for co-occurrence analysis.
* **`co_ocurence.py`**: Transforms a Term-Document Matrix (TDM) into a Jaccard similarity graph, enabling the evaluation of word associations and modularity within the corpus.
* **`categories_coocurence_window.py`**: Automated pipeline for group-based co-occurrence and community analysis.
* **`Bruge_share_on_graph.py`**: Calculating Bruges' term frequency share for Gephi visualization.
*  **`graph_evaluation.py`**: CMeasuring graph centrality and semantic distances to evaluate Bruges' competitive positioning.
* **`City_community_matrix.py`**: Detecting lexical communities to calculate city-specific weights based on Jaccard co-occurrence graphs.
#### 6. unused
* **`no_more_useful`**: Folder containing early experimental scripts (e.g. old TF‑IDF/LDA graph integrations, alternative clustering tests,Graph source_target_page) kept for transparency but not used in the final pipeline.
***

# 📊 Key Visualizations
##### [:rocket: Go to Contents Overview](#contents-overview)

### 1. Topic Distribution (LDA Heatmap)
Our model reveals that **Bruges** dedicates 54% of its content to *"Romantic heritage walks"*, a niche strategy compared to Amsterdam's functional focus.


<img width="552" height="288" alt="HEATMAP LDA" src="https://github.com/user-attachments/assets/9920a0e3-4097-42d8-b5a8-7a7dea7ffaf1" />


### 2. Lexical Signatures (WordClouds)
Visualizing the unique vocabulary of each zones after TF-IDF weighting.

<img width="689" height="197" alt="Words clouds north_south" src="https://github.com/user-attachments/assets/45bac0f8-21af-45f0-a065-dc9cb41a55bb" />
<img width="693" height="206" alt="Words clouds sea_nosea" src="https://github.com/user-attachments/assets/663a3110-eb87-44a4-8b03-1daa084634ad" />


### 3. Global Profiling (Hierarchical Clustering)
To visualize global similarities between cities, we applied hierarchical clustering based on average TF-IDF vectors.
* **Metric**: Cosine Distance (chosen to focus on discourse structure regardless of content volume).
* **Linkage**: Average Linkage (offering a balance between noise sensitivity and cluster stability).
<img width="479" height="234" alt="Dendogramme clusters" src="https://github.com/user-attachments/assets/de401aec-9e77-47b6-bece-d78890cb6f92" />


**Insights from the Dendrogram:**
* **Cluster 1 (Coastal)**: Ostend and Lisbon (Sea, port, and promenade themes).
* **Cluster 2 (Mediterranean/Urban)**: Barcelona, Valencia, Rome, and Manchester (Generalist profile: culture, shopping, and gastronomy).
* **Cluster 3 (Northern Historic)**: Amsterdam, Copenhagen, **Bruges**, and Cologne. Bruges sits within this group, sharing a lexical DNA centered on canals, history, and museums.

### 4. Semantic Zoom: Bigrams & Concordance
Beyond global trends, we analyzed how specific keywords are used in different contexts.
* **"Romantic"**: In **Bruges**, it refers to heritage and canals; in **Amsterdam**, it points to "romantic weekend" travel packages.
* **"Restaurant"**: Linked to "high gastronomy" in Barcelona, vs. "craft and authenticity" in Bruges.

### 5. Structural Network (Gephi)
Mapping the connectivity and content hubs of the analyzed tourist sites.
***

#### 5.1 Global Lexical Networks (Cosine & Jaccard)
To capture both shared structures and destination‑specific signatures, two complementary graphs were built from the same co‑occurrence data:

- **Cosine similarity graph** – reveals the common thematic backbone of European tourist communication (heritage, accessibility, urban atmosphere) and how cities cluster together in a shared semantic space.  
- **Jaccard similarity graph** – highlights each city’s distinctive lexical fingerprint by focusing on the presence/absence of terms, making local vocabularies and niche positioning (e.g. Bruges’ strongly place‑anchored lexicon) more visible.

**Global cosine similarity graph**
<br>
<img width="500" height="935" alt="image" src="https://github.com/user-attachments/assets/d579ce6c-27eb-4a38-9afa-ceba549b8711" />

**Global Jaccard similarity graph**
<br>
<img width="500" height="985" alt="image" src="https://github.com/user-attachments/assets/f08cac32-ba9f-49ec-a0ff-f0fcf9818aa7" />


#### 5.2 Bruges in the Shared Lexical Graph
This projection shows how **Bruges**’ key terms are embedded in the global co‑occurrence network built from all cities’ content. Nodes represent words, edges represent co‑occurrences within a sliding window, and colors indicate communities detected by modularity clustering.



**What this graph highlights:**<img width="502" height="379" alt="image" src="https://github.com/user-attachments/assets/9196ea2c-0e53-4d1d-80bf-ae970efd1924" />

- Bruges forms a **dense local community** of toponyms and heritage‑related terms (streets, canals, landmarks), confirming its strong local lexical identity.
- At the same time, several Bruges nodes connect to the **global backbone** of the network (e.g. culture, museums, restaurants), showing how the city plugs into the common European tourism narrative.

##### [:rocket: Go to Contents Overview](#contents-overview)

***

# 👥 ACE Click Team
##### [:rocket: Go to Contents Overview](#contents-overview)

* **Charlotte Fontaine** - [Contact](mailto:charlotte.fontaine@student.uclouvain.be)
* **Eva Martin** - [Contact](mailto:eva.martin@student.uclouvain.be)
* **Amélie Paulart** - [Contact](mailto:amélie.paulart@student.uclouvain.be)

***
<p align="center"><b>⭐ If you find our Web Mining analysis useful, please give it a star on GitHub!</b></p>
