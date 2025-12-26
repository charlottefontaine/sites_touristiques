
import os 
from bs4 import BeautifulSoup 
import requests
import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import re 
import time

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


def scrape_website(url, base_domain, site_name, max_links=5, homepage_sleep=3, link_sleep=2):
    """
    Scrape une page d'accueil et ses liens internes.
    Essaie d'abord requests + BeautifulSoup (rapide), fallback à Selenium si nécessaire.
    
    Args:
        url: URL de la page d'accueil à scraper
        base_domain: domaine de base pour filtrer les liens internes (ex: "barcelonaturisme.com")
        site_name: nom du site pour la sauvegarde (ex: "barcelone")
        max_links: nombre maximum de liens internes à scraper (défaut: 5)
        homepage_sleep: temps d'attente après le chargement de la homepage (défaut: 3)
        link_sleep: temps d'attente après le chargement de chaque lien (défaut: 2)
    
    Returns:
        dict: {"homepage_url": homepage_html, "link_url_1": html, "link_url_2": html, ...}
    """
    
    # 1. Essayer d'abord avec requests (rapide)
    print(f"📡 Tentative avec requests pour {site_name}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        homepage_html = response.text
        
        # Vérifier si le contenu est significatif (pas vide/minimal)
        if len(homepage_html) > 1000:
            soup = BeautifulSoup(homepage_html, 'html.parser')
            links_elements = soup.find_all('a', href=True)
            internal_links = [link['href'] for link in links_elements if base_domain in link['href']]
            
            if len(internal_links) > 0:
                print(f"✅ Succès avec requests ! {len(internal_links)} lien(s) trouvé(s)")
                
                # 2. Récupérer le HTML de chaque lien interne
                internal_htmls = {}
                for link in internal_links[:max_links]:
                    try:
                        response = requests.get(link, timeout=10)
                        internal_htmls[link] = response.text
                    except:
                        print(f"⚠️ Impossible de charger {link}")
                
                # 3. Sauvegarder les HTMLs
                save_html(homepage_html, f"output/{site_name}_homepage.html")
                for i, (link, html) in enumerate(internal_htmls.items(), 1):
                    save_html(html, f"output/{site_name}_link_{i}.html")
                
                return {url: homepage_html, **internal_htmls}
    except Exception as e:
        print(f"⚠️ Requests échoué ({e}), fallback à Selenium...")
    
    # Fallback : utiliser Selenium
    print(f"🔧 Utilisation de Selenium pour {site_name}...")
    driver = webdriver.Chrome(service=Service())
    
    try:
        driver.get(url)
        time.sleep(homepage_sleep)
        
        # 2. Récupérer le HTML de la homepage
        homepage_html = driver.page_source
        
        # 3. Extraire les liens internes
        links_elements = driver.find_elements(By.XPATH, '//a[@href]')
        internal_links = [link.get_attribute('href') for link in links_elements if base_domain in link.get_attribute('href')]
        
        # 4. Récupérer le HTML de chaque lien interne
        internal_htmls = {}
        for link in internal_links[:max_links]:
            driver.get(link)
            time.sleep(link_sleep)
            internal_htmls[link] = driver.page_source
        
        print(f"✅ Succès avec Selenium ! {len(internal_htmls)} lien(s) chargé(s)")
        
        # 5. Sauvegarder les HTMLs
        save_html(homepage_html, f"output/{site_name}_homepage.html")
        for i, (link, html) in enumerate(internal_htmls.items(), 1):
            save_html(html, f"output/{site_name}_link_{i}.html")
        
        # 6. Retourner le dictionnaire complet
        return {url: homepage_html, **internal_htmls}
        
    finally:
        driver.quit()