import requests
import os
def verifier_urls(dico_villes):
    """
    dico_villes : dict { 'Ville' : 'https://url...' }
    Vérifie le status code de chaque site et retourne un dictionnaire des résultats.
    """
    resultats = {}

    for ville, url in dico_villes.items():
        try:
            r = requests.get(url, timeout=10)
            status = r.status_code

            # Interprétation simple
            if status == 200:
                message = "OK - accessible"
            elif 300 <= status < 400:
                message = "Redirection"
            elif 400 <= status < 500:
                message = "Erreur client"
            elif 500 <= status < 600:
                message = "Erreur serveur"
            else:
                message = "Statut inattendu"

            resultats[ville] = {
                "url": url,
                "status_code": status,
                "etat": message
            }

        except requests.exceptions.RequestException as e:
            # En cas d’erreur (timeout, mauvaise URL…)
            resultats[ville] = {
                "url": url,
                "status_code": None,
                "etat": f"Erreur : {type(e).__name__}"
            }

    return resultats
cities = {
    "Barcelone": "https://www.barcelonaturisme.com/wv3/fr/",
    "Lisbonne": "https://www.visitlisboa.com/","Rome" : "https://www.rome.net/","Marseille" : "https://www.marseille-tourisme.com/", "Anvers" : "https://visit.antwerpen.be/fr"}

resultats = verifier_urls(cities)

for ville, infos in resultats.items():
    print(f"{ville} -> {infos}")


print("Répertoire courant :", os.getcwd())
def sauvegarder_homepages(dico_villes, dossier="pages_html"):
    """
    Télécharge le HTML brut de chaque page d'accueil 
    et l'enregistre dans un fichier local.

    dico_villes : dict -> { "Ville" : "URL" }
    dossier : nom du dossier où stocker les fichiers
    """

    # 1. Créer un dossier si non existant
    if not os.path.exists(dossier):
        os.makedirs(dossier)

    for ville, url in dico_villes.items():
        print(f" Téléchargement de la page pour {ville}...")

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Lève une erreur si statut ≠ 200

            # 2. Nettoyer nom de fichier (accents, espaces)
            nom_fichier = ville.replace(" ", "_").lower() + ".html"
            chemin = os.path.join(dossier, nom_fichier)

            # 3. Sauvegarder le HTML dans un fichier
            with open(chemin, "w", encoding="utf-8") as f:
                f.write(r.text)

            print(f"    Page enregistrée : {chemin}")

        except Exception as e:
            print(f"    Erreur pour {ville} : {e}")

if __name__ == "__main__":
    # Ton dictionnaire
    villes = {
        "Barcelone": "https://www.barcelonaturisme.com/wv3/fr/",
        "Lisbonne": "https://www.visitlisboa.com/",
        "Rome": "https://www.rome.net/",
        "Marseille": "https://www.marseille-tourisme.com/",
        "Anvers": "https://visit.antwerpen.be/fr",
        "Amsterdam": "https://www.iamsterdam.com/en",
        "Valence": "https://www.visitvalencia.com/fr",
        "Copenhagen": "https://www.visitcopenhagen.com/",
        "Manchester": "https://www.visitmanchester.com/",
        "Cologne": "https://www.cologne-tourism.com/"
    }

    # LANCEMENT DU SCRAPING
   
chemin_dossier = r"C:\Users\ameli\OneDrive - UCL\Master 1 UCL\Fucam\Web mining"
sauvegarder_homepages(villes, dossier=chemin_dossier)
  