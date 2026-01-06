"""
Generates wordclouds by area (SEA / NOSEA / NORTH / SOUTH)
from raw frequency TDMs aggregated by area.
Inputs
------
data/processed/tdm_zone_sea_freq.csv, ...

Outputs
-------
output/wordclouds/sea_cities.png, ...
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from categories_cleaning import build_zone_freq_tdm


OUTPUT_DIR = "data/text_analysis/wordclouds"


def generate_wordcloud_from_tdm(
    csv_path: str,
    title: str,
    top_n: int = 200,
    figsize=(12, 6),
    output_dir: str = OUTPUT_DIR,
):
    df = pd.read_csv(csv_path)

    if "city" in df.columns:
        df = df.drop(columns=["city"])
    df = df.select_dtypes(include=[np.number])

    word_frequencies = df.sum(axis=0).sort_values(ascending=False).head(top_n)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=top_n,
    ).generate_from_frequencies(word_frequencies.to_dict())

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)

    safe_title = title.replace(" ", "_").lower()
    out_path = os.path.join(output_dir, f"{safe_title}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Wordcloud saved to {out_path}")


def main():
    build_zone_freq_tdm()  # construit tdm_zone_*_freq.csv si besoin

    tdm_files = {
        "SEA cities": "data/processed/tdm_zone_sea_freq.csv",
        "NO SEA cities": "data/processed/tdm_zone_nosea_freq.csv",
        "NORTH cities": "data/processed/tdm_zone_north_freq.csv",
        "SOUTH cities": "data/processed/tdm_zone_south_freq.csv",
    }

    for title, path in tdm_files.items():
        if not os.path.exists(path):
            print(f"[WARNING] Missing TDM file for {title}: {path}")
            continue
        generate_wordcloud_from_tdm(
            csv_path=path,
            title=title,
            top_n=200,
            output_dir=OUTPUT_DIR,
        )

    print("4 wordclouds generated (or attempted).")


if __name__ == "__main__":
    main()
