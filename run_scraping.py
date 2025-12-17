from parameters import CITIES, REQUEST_TIMEOUT
from Utils import *
import os

BASE_OUTPUT = "data"

def main():

    # 1. Check accessibility
    access = check_website_accessibility(CITIES, REQUEST_TIMEOUT)

    pages_data = []

    for city, info in access.items():
        if not info["accessible"]:
            print(f" {city} not accessible")
            continue

        print(f" Processing {city}")

        homepage_html = download_html(info["url"])
        save_html(homepage_html, f"{BASE_OUTPUT}/html/homepages/{city}.html")

        # 2. Extract & filter technical links
        all_links = extract_links_from_html(homepage_html, info["url"])
        domain = urlparse(info["url"]).netloc
        depth1_links = filter_technical_links(all_links, domain)


        for link in depth1_links:
            try:
                html = download_html(link)
                save_html(html, f"{BASE_OUTPUT}/html/depth1/{city}_{hash(link)}.html")

                text = clean_html_and_extract_text(html)

                pages_data.append({
                    "city": city,
                    "url": link,
                    "depth": 1,
                    "text": text,
                    "word_count": len(text.split())
                })

            except Exception:
                continue

    df = build_dataframe(pages_data)
    os.makedirs(f"{BASE_OUTPUT}/processed", exist_ok=True)
    df.to_csv(f"{BASE_OUTPUT}/processed/corpus.csv", index=False)
    print("Corpus ready")


if __name__ == "__main__":
    main()