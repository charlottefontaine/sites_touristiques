
from scraper_barcelone import scrape_barcelone 

from utils import clean_html, save_html, combine_texts, safe_filename



def main(): 

    all_html = {} 

    # 1. Scraper Lisbonne (idem pour d'autres villes) 

    all_html.update(scrape_barcelone()) 

    # 2. Nettoyer et sauvegarder le texte 
    print("✅ Scraping terminé, nettoyage et sauvegarde des textes...")

    for key, html in all_html.items(): 

        cleaned_text = clean_html(html) 
        filename = safe_filename(key)  # <-- transforme "https://..." en nom valide

        save_html(cleaned_text, f"output/{key}.txt") 
    
    print("✅ Textes nettoyés et sauvegardés !")    
    print("=== DEBUG KEYS ===") 

    for k in all_html.keys(): 

        print(repr(k)) 

    print("=== END DEBUG ===") 


    for key, html in all_html.items(): 

        cleaned_text = clean_html(html) 

        filename = safe_filename(key)  # <-- transforme "https://..." en nom valide 

        save_html(cleaned_text, f"output/{filename}.txt")  
    
    print
        # 3. Combiner tous les textes en un corpus global 

    combine_texts("output", "output/corpus_combined.txt") 

    print("✅ Scraping et corpus combiné terminés !") 

 

if __name__ == "__main__": 
    main()