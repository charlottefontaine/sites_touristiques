import pandas as pd
import os
import csv

# 1. Définition dynamique des chemins
# Ce code trouve automatiquement le dossier 'processed'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TFIDF = os.path.join(BASE_DIR, "data", "processed", "df_tfidf_by_city.csv")
OUTPUT_NODES = os.path.join(BASE_DIR, "data", "processed", "gephi_nodes.csv")
OUTPUT_EDGES = os.path.join(BASE_DIR, "data", "processed", "gephi_edges.csv")

# Tes Topics LDA pour la correspondance
TOPIC_LABELS = {
    0: "Romantic City Walks & Heritage",
    1: "Transportation & Guided Sightseeing",
    2: "Cultural Discovery & Museum Passes",
    3: "Food, Wine & Local Experiences",
    4: "Tourist Information & Digital Platforms",
    5: "Entertainment, Shopping & Family Attractions"
}


# Enrichis ces listes avec les mots affichés par ton LDA_analysis.py
TOPIC_KEYWORDS = {
    0: ["bruges", "canal", "walk", "heritage", "romantic", "history", "belfry", "medieval", "architecture", "old", "city", "stone"],
    1: ["airport", "bus", "shuttle", "station", "tour", "guide", "transport", "shuttle", "transfer", "line", "coach", "arrival"],
    2: ["museum", "art", "culture", "ticket", "pass", "exhibition", "gallery", "collection", "masterpiece", "history", "entry"],
    3: ["wine", "food", "restaurant", "local", "tasting", "gastronomy", "beer", "chocolate", "dinner", "cuisine", "specialty"],
    4: ["website", "online", "app", "information", "contact", "digital", "booking", "service", "mobile", "account", "login"],
    5: ["shopping", "theatre", "opera", "family", "kids", "attraction", "cinema", "park", "show", "performance", "store"]
}

def create_gephi_files():
    if not os.path.exists(INPUT_TFIDF):
        print(f"ERREUR : Le fichier TF-IDF est introuvable ici : {INPUT_TFIDF}")
        return

    # Charger la matrice
    df = pd.read_csv(INPUT_TFIDF, index_col=0)
    top_words = df.sum().sort_values(ascending=False).head(100).index
    df_reduced = df[top_words]

    # --- NODES ---
    nodes = []
    for word in top_words:
        assigned_topic = "Other"
        for tid, keywords in TOPIC_KEYWORDS.items():
            if any(k in word.lower() for k in keywords):
                assigned_topic = TOPIC_LABELS[tid]
                break
        nodes.append({'Id': word, 'Label': word, 'Topic': assigned_topic})
    
    pd.DataFrame(nodes).to_csv(OUTPUT_NODES, index=False, sep=';', quoting=csv.QUOTE_ALL)

    
# --- EDGES (Correction) ---
    co_matrix = df_reduced.T.dot(df_reduced)
    edges = []
    cols = co_matrix.columns
    
    # Option A : Baisse le seuil à 5% au lieu de 15%
    threshold = co_matrix.values.max() * 0.05 
    
    # Option B (Plus robuste) : On prend les 300 liens les plus forts
    # Cela garantit d'avoir un beau graphe dense sans être illisible
    all_edges = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            weight = co_matrix.iloc[i, j]
            if weight > 0:
                all_edges.append({'Source': cols[i], 'Target': cols[j], 'Weight': weight})
    
    # Tri et sélection des 300 meilleurs
    edges_df = pd.DataFrame(all_edges).sort_values(by='Weight', ascending=False).head(300)
    edges_df.to_csv(OUTPUT_EDGES, index=False, sep=';')



    print(f"\n--- SUCCÈS ---")
    print(f"Fichier Nodes créé : {OUTPUT_NODES}")
    print(f"Fichier Edges créé : {OUTPUT_EDGES}")
    print("Ouvre ton explorateur de fichiers dans 'data/processed' pour les voir.")

if __name__ == "__main__":
    create_gephi_files()