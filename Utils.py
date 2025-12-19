import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd


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
