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
- [💡 Strategic Recommendations](#-strategic-recommendations)
- [👥 ACE Click Team](#-ace-click-team)

***

# 🚀 Project Overview
##### [:rocket: Go to Contents Overview](#contents-overview)

In a European market welcoming over 340 million international tourists, official websites have become strategic digital storefronts. This project, led by the **ACE Click** consultancy, benchmarks **10 major European destinations** to identify digital best practices.

**The Challenge:** How do European tourist sites structure and promote their content, and what levers can **Bruges** use to strengthen its digital appeal?

**Cities Analyzed:** Barcelona, Lisbon, Rome, Ostend, Amsterdam, Valencia, Copenhagen, Manchester, Cologne, and **Bruges**.

***

# 🛠 Methodology & Pipeline
##### [:rocket: Go to Contents Overview](#contents-overview)

### 1. Data Collection (Web Scraping)
- **Strategy**: "Wide-but-shallow" crawling (Depth 1) to capture the most strategic content directly highlighted by tourism boards.
- **Corpus**: 894 documents collected from homepages and primary internal links.
- **Tools**: `BeautifulSoup`, `Requests`, `Selenium`.

### 2. NLP & Text Mining
- **Preprocessing**: Normalization of social media, removal of project-specific stopwords, and advanced **Lemmatization**.
- **Vectorization**: **TF-IDF** was chosen for its transparency, allowing us to isolate the "lexical DNA" of each city without the "black box" effect of neural models.
- **Topic Modeling**: Implementation of **Latent Dirichlet Allocation (LDA)** with $k=6$ topics to group content into editorial pillars.

### 3. Structural Analysis
- **Link Analysis**: Studying site architecture and page hierarchy.
- **Network Mapping**: Exporting co-occurrence data to **Gephi** for structural visualization.

***

# 📂 Repository Structure
##### [:rocket: Go to Contents Overview](#contents-overview)

* **`parameters.py`**: Maps cities to their tourism websites and categorizes them by geography and coastal access.
* **`Utils.py`**: Tools to scrape tourism websites and preprocess the text for NLP analysis.
* **`run_scrapping.py`**: Automated harvesting script extracting editorial content from the 10 target cities' tourism websites into a CSV corpus.
* **`corpus_analysis.py`**: Converts the corpus to JSON and generates an Excel report containing detailed statistical metrics on word counts and page distributions per city.
* **`corpus_cleaning.py`**: Cleans the text and generates TF-IDF visualizations to identify top keywords across city categories.
* **`corpus_check_post_cleaning.py`**: Performs quality control and statistical diagnostics on the cleaned matrix, including sparsity checks and Zipf's Law validation.
* **`compare_lemmatization_stemming.py`**: Compares lemmatization and stemming methods to evaluate their impact on vocabulary size and matrix sparsity.
* * **`categories_cleaning.py`**: Computes TF-IDF scores and splits the term-document matrix into geographical subsets (North, South, Sea, and No-Sea) for comparative analysis.
* **`word_clouds.py`**: Generates visual word clouds for each city category to highlight the most frequent terms across geographical zones.
* **`LDA_analysis.py`**: Performs Latent Dirichlet Allocation to discover thematic topics across the corpus and visualizes their distribution per city using heatmaps.
* **`Hyperparams_optimization_hierarchical_clust.py`**: Executes hierarchical clustering and generates dendrograms to visualize city similarities based on diverse distance metrics and linkage methods.
* **`hierarchical clustering and similarity analysis.py`**: Maps city similarities using cosine heatmaps and hierarchical dendrograms based on TF-IDF profiles.
* **`Graph source_target_page.py`**: Maps internal link structures to calculate PageRank and centralities, exporting the results for network visualization in Gephi.
* **`N-grams_analysis.py`**: Extracts and filters word pairs (bigrams) associated with specific tourism themes like "shopping" or "romantic" to identify local strategic trends.
* **`network_matrix.py`**: Maps TF-IDF word co-occurrences and LDA topics to node/edge files while calculating network metrics for visualization in Gephi.
* **`coocurence_window.py`**: Builds a sparse co-occurrence matrix using a sliding window to generate Jaccard or Cosine similarity graphs and identifies semantic clusters.
* **`concordance_analysis.py`**: Generates Keyword-in-Context (KWIC) tables to analyze how specific terms like "romantic" or "shopping" are used across different city subsets.
* **`co_ocurence.py`**: Transforms a Term-Document Matrix (TDM) into a Jaccard similarity graph, enabling the evaluation of word associations and modularity within the corpus.
* **`clean_tokens_cooc.py`**: Filters out CSS/HTML technical artifacts, multilingual noise, and generic tourism stopwords while lemmatizing English tokens to prepare a high-quality corpus for co-occurrence analysis.

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

# 💡 Strategic Recommendations
##### [:rocket: Go to Contents Overview](#contents-overview)

***

# 👥 ACE Click Team
##### [:rocket: Go to Contents Overview](#contents-overview)

* **Charlotte Fontaine** - [Contact](mailto:charlotte.fontaine@student.uclouvain.be)
* **Eva Martin** - [Contact](mailto:eva.martin@student.uclouvain.be)
* **Amélie Paulart** - [Contact](mailto:amélie.paulart@student.uclouvain.be)

***
<p align="center"><b>⭐ If you find our Web Mining analysis useful, please give it a star on GitHub!</b></p>
