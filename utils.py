
import os 
from bs4 import BeautifulSoup 
import requests
import selenium.webdriver as webdriver
import re 


from urllib.parse import urlparse 



def safe_filename(url): 

    """ 

    Transforme une URL en un nom de fichier valide sur Windows. 

    Exemple : "https://www.visitlisboa.com/" → "www.visitlisboa.com" 

    """ 

    parsed = urlparse(url) 

    # garder juste le netloc + path, remplacer "/" par "_" 

    name = parsed.netloc + parsed.path 

    name = name.strip("/")  # enlever / au début/fin 

    # remplacer tous les caractères interdits par "_" 

    name = re.sub(r'[\\/*?:"<>|]', "_", name) 

    # si vide, mettre un nom générique 

    if not name: 

        name = "site" 

    return name 


 

def scrape_homepage(url):
    response = requests.get(url)
    soup = BeautifulSoup.BeautifulSoup(response.text, 'html.parser')
    return soup
 

def save_html(html_content, filepath): 

    os.makedirs(os.path.dirname(filepath), exist_ok=True) 

    with open(filepath, "w", encoding="utf-8") as f: 

        f.write(html_content) 

 

def clean_html(html_content): 

    """Supprime scripts, styles et retourne uniquement le texte visible""" 

    soup = BeautifulSoup(html_content, "html.parser") 

 

    # supprimer scripts et styles 

    for script in soup(["script", "style"]): 

        script.decompose() 

 

    text = soup.get_text(separator=" ", strip=True) 

    return text 

 

def combine_texts(input_folder, output_file): 

    """Combine tous les fichiers txt dans un corpus unique""" 

    import glob 

    all_texts = [] 

    for file in glob.glob(f"{input_folder}/*.txt"): 

        with open(file, "r", encoding="utf-8") as f: 

            all_texts.append(f.read()) 

    with open(output_file, "w", encoding="utf-8") as f: 

        f.write("\n\n".join(all_texts)) 

print(f"Corpus combiné sauvegardé")