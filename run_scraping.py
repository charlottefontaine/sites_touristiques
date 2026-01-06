"""
Scrapes tourism websites for all configured cities and builds the initial text corpus.

The script:
- reads the list of cities and their URLs from the project parameters,
- checks website accessibility for each city (with a request timeout),
- downloads and saves the homepage HTML and depth‑1 pages for accessible cities,
- cleans HTML pages to extract main text content,
- builds a structured corpus (one row per page) and saves it as corpus.csv.

Inputs
------
- parameters.CITIES
    Dictionary of city names and associated homepage URLs.
- parameters.REQUEST_TIMEOUT
    Timeout (in seconds) for HTTP requests.
- Utils helpers
    check_website_accessibility(cities, timeout)
    download_html(url)
    save_html(html, path)
    extract_links_from_html(html, base_url)
    filter_technical_links(links, domain)
    clean_html_and_extract_text(html)
    build_dataframe(pages_data)

Outputs
-------
- HTML files:
    data/raw/html/homepages/{city}.html
    data/raw/html/depth1/{city}_{hash(url)}.html
- CSV corpus:
    data/processed/corpus.csv
    Columns typically include: city, url, depth, text, word_count.
"""
from parameters import CITIES, REQUEST_TIMEOUT
from Utils import *
import os

BASE_DIR = "data"
RAW_HTML_DIR = os.path.join(BASE_DIR, "raw", "html")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(os.path.join(RAW_HTML_DIR, "homepages"), exist_ok=True)
os.makedirs(os.path.join(RAW_HTML_DIR, "depth1"), exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    """
    Scrape homepages and depth‑1 pages for all cities and build corpus.csv.
    """
    # 1. Check accessibility
    access = check_website_accessibility(CITIES, REQUEST_TIMEOUT)

    pages_data = []

    for city, info in access.items():
        if not info["accessible"]:
            print(f" {city} not accessible")
            continue

        print(f" Processing {city}")

        # 2. Download & save homepage
        homepage_html = download_html(info["url"])
        save_html(
            homepage_html,
            os.path.join(RAW_HTML_DIR, "homepages", f"{city}.html"),
        )
                  
        # 3. Extract & filter technical links
        all_links = extract_links_from_html(homepage_html, info["url"])
        domain = urlparse(info["url"]).netloc
        depth1_links = filter_technical_links(all_links, domain)

        # 4. Visit each depth‑1 page
        for link in depth1_links:
            try:
                html = download_html(link)
                save_html(
                    html,
                    os.path.join(RAW_HTML_DIR, "depth1", f"{city}_{hash(link)}.html"),
                )

                text = clean_html_and_extract_text(html)

                pages_data.append(
                    {
                        "city": city,
                        "url": link,
                        "depth": 1,
                        "text": text,
                        "word_count": len(text.split()),
                    }
                )

            except Exception:
                continue

    # 5. Build final corpus
    df = build_dataframe(pages_data)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    corpus_path = os.path.join(PROCESSED_DIR, "corpus.csv")
    df.to_csv(corpus_path, index=False)
    print(f"Corpus saved to {corpus_path}")

if __name__ == "__main__":
    main()