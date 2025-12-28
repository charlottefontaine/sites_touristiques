import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import numpy as np

def generate_wordcloud_from_tdm(csv_path,title,top_n=200,figsize=(12, 6),output_dir="output"):

    df = pd.read_csv(csv_path)
    if "city" in df.columns:
        df = df.drop(columns=["city"])

    df = df.select_dtypes(include=[np.number])
    word_frequencies = (df.sum(axis=0).sort_values(ascending=False).head(top_n))

    wordcloud = WordCloud(width=1000,height=500,background_color="white",max_words=top_n).generate_from_frequencies(word_frequencies.to_dict())

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.replace(" ", "_").lower()
    plt.savefig(f"{output_dir}/{safe_title}.png")
    plt.close()

tdm_files = {
    "SEA cities": "data/processed/tdm_sea.csv",
    "NO SEA cities": "data/processed/tdm_nosea.csv",
    "NORTH cities": "data/processed/tdm_north.csv",
    "SOUTH cities": "data/processed/tdm_south.csv"
}

for title, path in tdm_files.items():
    generate_wordcloud_from_tdm(csv_path=path,title=title,top_n=200,output_dir="output/wordclouds")

print("4 wordclouds generated successfully")
