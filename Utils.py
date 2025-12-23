import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import json
import nltk
from collections import Counter
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



#---------------------------------------------------
# Part1. WEB SCRAPING UTILITIES
#--------------------------------------------------

# 1. CHECK WEBSITE ACCESSIBILITY
def check_website_accessibility(cities: dict, timeout: int = 10) -> dict:
    """
    Input:
        cities: dict {city_name: homepage_url}
    Output:
        dict {city: {url, status_code, accessible}}
    """
    results = {}

    for city, url in cities.items():
        try:
            r = requests.get(url, timeout=timeout)
            results[city] = {
                "url": url,
                "status_code": r.status_code,
                "accessible": r.status_code == 200
            }
        except requests.RequestException:
            results[city] = {
                "url": url,
                "status_code": None,
                "accessible": False
            }

    return results

# 2. DOWNLOAD HTML PAGE
def download_html(url: str, timeout: int = 10) -> str:
    """
    Input:
        url: page URL
    Output:
        raw HTML content (string)
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# 3. SAVE HTML LOCALLY
def save_html(content: str, filepath: str) -> None:
    """
    Input:
        content: HTML content
        filepath: local path
    Output:
        None (writes file)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# 4. EXTRACT ALL LINKS FROM HOMEPAGE
def extract_links_from_html(html: str, base_url: str) -> list:
    """
    Input:
        html: raw HTML
        base_url: homepage URL
    Output:
        list of absolute URLs
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        absolute_url = urljoin(base_url, a["href"])
        links.append(absolute_url)

    return list(set(links))

# 5. TECHNICAL LINK FILTERING
def filter_technical_links(links: list, base_domain: str) -> list:
    """
    Removes technical, legal, external, and non-editorial links.
    Keeps only internal HTML pages potentially relevant for content analysis.
    """
    excluded_keywords = [
        "privacy", "cookie", "legal", "terms",
        "press", "media", "partner",
        "login", "account", "subscribe",
        "job", "career",
        "accessibility", "newsletter"
    ]

    excluded_extensions = (
        ".pdf", ".jpg", ".jpeg", ".png", ".svg",
        ".mp4", ".zip"
    )

    filtered_links = []

    for link in links:
        link_lower = link.lower()
        parsed = urlparse(link)

        # External links
        if parsed.netloc and base_domain not in parsed.netloc:
            continue

        # Anchors, mail, phone
        if link_lower.startswith(("mailto:", "tel:")):
            continue
        if "#" in link_lower:
            continue

        # File extensions
        if link_lower.endswith(excluded_extensions):
            continue

        # Tracking parameters
        if any(param in link_lower for param in ["utm_", "fbclid", "ref="]):
            continue

        # Non-editorial keywords
        if any(keyword in link_lower for keyword in excluded_keywords):
            continue

        filtered_links.append(link)

    return list(set(filtered_links))

# 6. CLEAN HTML AND EXTRACT TEXT
def clean_html_and_extract_text(html: str) -> str:
    """
    Removes scripts, styles and returns visible text.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    return soup.get_text(separator=" ", strip=True)

# 7. BUILD FINAL DATASET
# --------------------------------------------------
def build_dataframe(pages: list) -> pd.DataFrame:
    """
    Input:
        pages: list of dicts with page metadata
    Output:
        pandas DataFrame ready for CSV export
    """
    return pd.DataFrame(pages)

#---------------------------------------------------
# Part2. DATA PROCESSING UTILITIES
#--------------------------------------------------

def csv_to_json(csv_path, json_path):
    """
    Convert CSV file to JSON, it allows us to reuse the corpus in different formats
    without rerunning the scraping process.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"CSV successfully converted: {json_path}")
    return df  


#---------------------------------------------------
# Part3. CLEANING PHASE BEFORE NLP PROCESSING
#---------------------------------------------------


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens


def tokenize_json_by_city_url(json_path):
    """
    Input:
        json_path: path to JSON file
    Output:
        dict with key = (city, url) and value = list of tokens
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dict_tokens = {}

    for item in data:
        if "text" in item and "city" in item and "url" in item:
            key = (item["city"], item["url"])
            tokens = tokenize_text(item["text"])
            dict_tokens[key] = tokens

    return dict_tokens


def build_term_document_matrix(documents):
    """
    documents : dict
        key   -> document name (str)
        value -> list of words (list[str])
    """
    vocabulary = sorted(set(
        word for words in documents.values() for word in words
    ))

    data = {}
    for doc_name, words in documents.items():
        counter = Counter(words)
        data[doc_name] = [counter.get(word, 0) for word in vocabulary]

    return pd.DataFrame.from_dict(
        data, orient="index", columns=vocabulary
    )


def clean_term_document_matrix(matrix, corrector=None, keep_numbers=False):
    """
    matrix : pd.DataFrame
        term-document matrix (rows = documents, columns = terms)
    corrector : function or None
        optional spell checker
    keep_numbers : bool
        keep terms containing digits if True
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    new_columns = {}

    for term in matrix.columns:
        term_clean = term.lower()
        term_clean = emoji_pattern.sub("", term_clean)
        term_clean = term_clean.replace("\n", " ")  
        term_clean = term_clean.translate(str.maketrans("", "", string.punctuation + "€'’‘“”"))

        if not keep_numbers and re.search(r"\d", term_clean):
            continue

        if term_clean.strip() == "":
            continue

        if corrector is not None:
            term_clean = corrector(term_clean)

        if term_clean in new_columns:
            new_columns[term_clean] += matrix[term]
        else:
            new_columns[term_clean] = matrix[term].copy()

    return pd.DataFrame(new_columns, index=matrix.index)


def normalize_social_media_terms(term_document_matrix):
    """
    Normalize concatenated social media strings into
    generic platform tokens (instagram, facebook, etc.)
    """

    platforms = {
        "instagram": r"instagram",
        "facebook": r"facebook",
        "twitter": r"(twitter|xcom)",
        "youtube": r"youtube",
        "linkedin": r"linkedin",
        "tiktok": r"tiktok"
    }

    new_columns = {}

    for term in term_document_matrix.columns:
        normalized_term = term.lower().strip(", ")

        for platform, pattern in platforms.items():
            if re.search(pattern, normalized_term):
                normalized_term = platform
                break

        if normalized_term in new_columns:
            new_columns[normalized_term] += term_document_matrix[term]
        else:
            new_columns[normalized_term] = term_document_matrix[term].copy()

    return pd.DataFrame(new_columns, index=term_document_matrix.index)



def remove_nltk_stopwords(matrix, language="english"):
    """
    Remove standard NLTK stopwords from matrix
    """
    try:
        stopwords_nltk = set(stopwords.words(language))
    except LookupError:
        nltk.download("stopwords")
        stopwords_nltk = set(stopwords.words(language))

    columns_to_keep = [col for col in matrix.columns if col.lower() not in stopwords_nltk]

    return matrix[columns_to_keep]


def remove_project_stopwords(matrix, project_stopwords):
    """
    Remove project-specific stopwords

    Parameters
    ----------
    matrix : pd.DataFrame
        term-document matrix (columns = terms)
    project_stopwords : list or set
        list of project-specific stopwords

    Returns
    -------
    pd.DataFrame
        cleaned matrix
    """
    project_stopwords = {sw.lower().strip() for sw in project_stopwords}

    columns_to_drop = [col for col in matrix.columns if col.lower() in project_stopwords]

    return matrix.drop(columns=columns_to_drop, errors="ignore")


def lemmatize_matrix_nltk(matrix):
    """
    Lemmatize the term-document matrix
    """
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        lemmatizer = WordNetLemmatizer()

    new_columns = {}

    for term in matrix.columns:
        lemma = lemmatizer.lemmatize(term)
        if lemma in new_columns:
            new_columns[lemma] += matrix[term]
        else:
            new_columns[lemma] = matrix[term].copy()

    return pd.DataFrame(new_columns, index=matrix.index)


def filter_terms_by_frequency(matrix, min_df=4, max_df_ratio=1.0):
    """
    Filter terms by document frequency

    matrix : pd.DataFrame
        term-document matrix
    min_df : int
        minimum number of documents a term must appear in
    max_df_ratio : float (0 < max_df_ratio <= 1)
        maximum proportion of documents a term can appear in
    """
    n_documents = matrix.shape[0]
    freq_doc = (matrix > 0).sum(axis=0)

    columns_to_keep = [
        term for term in matrix.columns
        if freq_doc[term] >= min_df
        and freq_doc[term] <= max_df_ratio * n_documents
    ]

    return matrix[columns_to_keep]

def remove_miniwords(matrix, min_length=2):
    """
    Remove terms with length less than min_length

    Parameters
    ----------
    matrix : pd.DataFrame
        term-document matrix (columns = terms)
    min_length : int
        minimum length of terms to keep

    Returns
    -------
    pd.DataFrame
        cleaned matrix
    """
    columns_to_drop = [col for col in matrix.columns if len(col) < min_length]

    return matrix.drop(columns=columns_to_drop, errors="ignore")




print("finally")