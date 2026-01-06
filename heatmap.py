
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import des fonctions existantes (assurez-vous que ces fichiers sont dans le même dossier)
from Utils import *  # On a juste besoin de la fonction de base ici
from parameters import *

# ---------------------------------------------------------
# 1. Définition des Villes et Zones (Mettre à jour si nécessaire)
# ---------------------------------------------------------
SEA_CITIES = {"Barcelona", "Lisbon", "Copenhagen", "Ostend", "Valencia"}
NOSEA_CITIES = {"Rome", "Manchester", "Cologne", "Amsterdam", "Bruges"}
SOUTH_CITIES = {"Barcelona", "Lisbon", "Rome", "Valencia"}
NORTH_CITIES = {"Amsterdam", "Copenhagen", "Manchester", "Cologne", "Ostend", "Bruges"}

# ---------------------------------------------------------
# 2. Fonctions Utilitaires Locales
# ---------------------------------------------------------
def main(dict_tokens):
    cities = set(key[0] for key in dict_tokens.keys())
    print(f"Cities present: {sorted(cities)}")
    print(f"Total entries: {len(dict_tokens)}")

    df_tdm = build_term_document_matrix(dict_tokens)
    print(f"Term-document matrix shape: {df_tdm.shape}")

    df_tdm = clean_term_document_matrix(df_tdm, None, keep_numbers=False)
    df_tdm = normalize_social_media_terms(df_tdm)
    df_tdm = remove_nltk_stopwords(df_tdm)
    df_tdm = lemmatize_matrix_nltk(df_tdm)
    df_tdm = remove_project_stopwords(df_tdm, project_stopwords)
    df_tdm = remove_miniwords(df_tdm, min_length=3)

    # Frequency filtering
    df_freq_terms = filter_terms_by_frequency(df_tdm)
    print(f"Final shape after filtering: {df_freq_terms.shape}")

    # Add city column
    df_freq_terms = df_freq_terms.copy()
    df_freq_terms["city"] = [t[0] for t in df_freq_terms.index]
    cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"]
    df_freq_terms = df_freq_terms[cols]

    os.makedirs("data/processed", exist_ok=True)
    return df_freq_terms


def split_corpus_by_city_criteria(corpus, sea_cities, nosea_cities, south_cities, north_cities):
    """Divise le corpus (liste de dicts) en 4 sous-corpus."""
    split_data = {
        "SEA": [],
        "NOSEA": [],
        "SOUTH": [],
        "NORTH": []
    }

    for doc in corpus:
        city = doc.get("city")
        if not city:
            continue

        if city in sea_cities:
            split_data["SEA"].append(doc)
        if city in nosea_cities:
            split_data["NOSEA"].append(doc)
        if city in south_cities:
            split_data["SOUTH"].append(doc)
        if city in north_cities:
            split_data["NORTH"].append(doc)

    return split_data

def tokenize_data_in_memory(data_list):
    """
    Remplace 'tokenize_json_by_city_url' pour les données déjà chargées en mémoire.
    Convertit une liste de documents bruts en dictionnaire de tokens.
    """
    dict_tokens = {}
    for item in data_list:
        if "text" in item and "city" in item and "url" in item:
            key = (item["city"], item["url"])
            # On utilise la fonction tokenize_text importée de Utils
            tokens = tokenize_text(item["text"])
            dict_tokens[key] = tokens
    return dict_tokens

def plot_heatmap_top_words(df_freq_terms, zone_name, top_n=15, output_folder="data/processed/plots"):
    """Génère et sauvegarde la heatmap."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Identifier les colonnes de mots (tout sauf 'city')
    terms_only = [c for c in df_freq_terms.columns if c != "city"]
    
    if not terms_only:
        print(f"⚠️ Attention : Aucun mot trouvé pour la zone {zone_name}. Pas de heatmap générée.")
        return

    # Trouver les top N mots globaux pour cette zone
    top_words_zone = df_freq_terms[terms_only].sum().sort_values(ascending=False).head(top_n).index
    
    # Agréger par ville pour la heatmap
    df_heatmap = df_freq_terms.groupby("city")[top_words_zone].sum()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap, annot=True, cmap="YlOrRd", fmt=".0f")
    plt.title(f"Heatmap des {top_n} mots les plus fréquents - Zone: {zone_name}")
    plt.xlabel("Mots")
    plt.ylabel("Villes")
    plt.tight_layout()
    
    filename = f"{output_folder}/heatmap_{zone_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Heatmap créée : {filename}")

# ---------------------------------------------------------
# 3. Exécution Principale
# ---------------------------------------------------------

if __name__ == "__main__":
    path_to_full_corpus = "data/processed/corpus_json.json"
    
    print("1. Chargement du fichier JSON global...")
    try:
        with open(path_to_full_corpus, "r", encoding="utf-8") as f:
            full_corpus_data = json.load(f) # <--- C'est ici qu'on charge les données !
    except FileNotFoundError:
        print(f"Erreur : Le fichier {path_to_full_corpus} est introuvable.")
        exit()

    print(f"   {len(full_corpus_data)} documents chargés.")

    print("2. Création des corpus par zone...")
    corpora = split_corpus_by_city_criteria(
        corpus=full_corpus_data, # On passe les données, pas le chemin
        sea_cities=SEA_CITIES,
        nosea_cities=NOSEA_CITIES,
        south_cities=SOUTH_CITIES,
        north_cities=NORTH_CITIES
    )

    # Boucle sur chaque zone pour traiter et créer la heatmap
    for zone, data_docs in corpora.items():
        print(f"\n--- Traitement de la zone : {zone} ({len(data_docs)} docs) ---")
        
        if len(data_docs) == 0:
            print(f"⚠️ Aucun document pour la zone {zone}, on passe.")
            continue

        # A. Tokenisation (Version en mémoire)
        print("   Tokenisation...")
        dict_tokens = tokenize_data_in_memory(data_docs)

        # B. Nettoyage et Matrice (Utilisation de votre fonction main importée)
        # Cette fonction fait : Matrix -> Clean -> Norm -> Stopwords -> Freq Filter -> Ajout col 'city'
        print("   Nettoyage et création matrice...")
        df_processed = main(dict_tokens)

        # C. Génération Heatmap
        print("   Génération Heatmap...")
        plot_heatmap_top_words(df_processed, zone, top_n=15, output_folder="data/processed/plots")

    print("\n🎉 Terminé ! Toutes les heatmaps sont à jour.")
