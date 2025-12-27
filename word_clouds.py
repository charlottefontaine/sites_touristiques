import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import os
import numpy as np

def generate_wordcloud_from_tdm(
    csv_path,
    title,
    top_n=200,
    figsize=(12, 6),
    output_dir="output"
    ):

    df = pd.read_csv(csv_path)

    # Remove non-term columns if present
    if "city" in df.columns:
        df = df.drop(columns=["city"])

    # Select only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Compute global term frequencies
    word_frequencies = df.sum(axis=0)

    # Keep top N terms
    word_frequencies = (
        word_frequencies
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Generate word cloud
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=top_n
    ).generate_from_frequencies(word_frequencies.to_dict())

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(f"{output_dir}/{safe_title}.png")
    plt.close()

tdm_files = {
    "Global corpus": "data/processed/df_freq_terms.csv",
    "SEA cities": "data/processed/tdm_sea.csv",
    "NO SEA cities": "data/processed/tdm_nosea.csv",
    "SOUTH cities": "data/processed/tdm_south.csv",
    "NORTH cities": "data/processed/tdm_north.csv"
}

for title, path in tdm_files.items():
    generate_wordcloud_from_tdm(
        csv_path=path,
        title=title,
        top_n=200
    )

print("Word clouds generated and saved.")