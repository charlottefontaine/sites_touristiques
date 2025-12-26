
# scraper_lisbon.py 

from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from utils import save_html 
import time 

 

def scrape_barcelone(): 

    url = "https://www.barcelonaturisme.com/wv3/fr/" 

    # 1. Lancer Chrome (assurez-vous que chromedriver est installé) 

    driver = webdriver.Chrome(service=Service()) 
    driver.get(url) 
    time.sleep(3)  # attendre que la page charge 

    # 2. Récupérer le HTML de la homepage 

    homepage_html = driver.page_source 

    # 3. Extraire les liens internes (menu / sections) 

    links_elements = driver.find_elements(By.XPATH, '//a[@href]') 
    internal_links = [link.get_attribute('href') for link in links_elements if url in link.get_attribute('href')] 

    # 4. Optionnel : récupérer le HTML de chaque lien interne 

    internal_htmls = {} 
    for link in internal_links[:5]:  # limiter à 5 pour test 
        driver.get(link) 
        time.sleep(2) 
        internal_htmls[link] = driver.page_source 


    driver.quit() 


    # 5. Sauvegarder HTML 

    save_html(homepage_html, "output/barcelone_homepage.html") 
    for i, (link, html) in enumerate(internal_htmls.items(), 1): 
        save_html(html, f"output/barcelone_link_{i}.html") 


    # 6. Retourner dictionnaire pour main.py 

    return {"barcelone": homepage_html, **internal_htmls} 

