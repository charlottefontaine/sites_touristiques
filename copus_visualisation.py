import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os


SEA_CITIES = {"Barcelona", "Lisbon", "Copenhagen", "Ostend", "Valencia"}
NOSEA_CITIES = {"Rome", "Manchester", "Cologne", "Amsterdam", "Bruges"}
NORTH_CITIES = {"Amsterdam", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"}
SOUTH_CITIES = {"Barcelona", "Lisbon", "Rome", "Valencia"}

def get_zone(city):
    if city in SEA_CITIES:
        return "SEA"
    if city in NOSEA_CITIES:
        return "NOSEA"
    if city in NORTH_CITIES:
        return "NORTH"
    if city in SOUTH_CITIES:
        return "SOUTH"
    return "Other"

# ---------------
# Heatmap excel
# ---------------
def save_heatmap_excel(df, output_path):
    wb = load_workbook(output_path)
    ws = wb.active
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        print("no numeric data")
        return

    max_val = df_numeric.to_numpy().max()

    for row in ws.iter_rows(min_row=2, min_col=2):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                intensity = int(255 * cell.value / max_val) if max_val else 0
                color = f"FF{255-intensity:02X}{255-intensity:02X}"
                cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")

    wb.save(output_path)


df = pd.read_csv("data/processed/df_freq_terms_all.csv")
df["Zone"] = df["city"].apply(get_zone)
terms = [c for c in df.columns if c not in ["city", "Zone"]]
df[terms] = df[terms].apply(pd.to_numeric, errors='coerce').fillna(0)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# PART 1 : Word Heatmap
# -----------------------------
def word_heatmap(data, name):
    freq = data[terms].sum().sort_values(ascending=False).head(10)
    df_hm = data.groupby("city")[freq.index].sum()
    path = f"data/processed/heatmap_words_{name}.xlsx"
    df_hm.to_excel(path)
    save_heatmap_excel(df_hm, path)

word_heatmap(df, "GLOBAL")
for zone in ["SEA", "NOSEA", "NORTH", "SOUTH"]:
    word_heatmap(df[df["Zone"] == zone], zone)

# ---------------------------------
# PART 2 : Heatmap for topics (LDA)
# ---------------------------------
tdm = df[terms].values
cities = df["city"].values

lda = LatentDirichletAllocation(n_components=6, random_state=42)
doc_topic = lda.fit_transform(tdm)

df_topics = pd.DataFrame(doc_topic, columns=[f"Topic_{i}" for i in range(6)])
df_topics["city"] = cities
df_topics["Zone"] = df_topics["city"].apply(get_zone)

def topic_heatmap(data, name):
    df_hm = data.groupby("city").mean()
    df_hm = df_hm.div(df_hm.sum(axis=1), axis=0)
    path = f"data/processed/heatmap_topics_{name}.xlsx"
    df_hm.to_excel(path)
    save_heatmap_excel(df_hm, path)

topic_heatmap(df_topics.drop(columns="Zone"), "GLOBAL")
for zone in ["SEA", "NOSEA", "NORTH", "SOUTH"]:
    topic_heatmap(df_topics[df_topics["Zone"] == zone].drop(columns="Zone"), zone)

print("Finally")
