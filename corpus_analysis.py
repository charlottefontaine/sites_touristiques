"""
    Build basic descriptive tables on the raw corpus and export them to Excel.
    Inputs : 
    data/processed/corpus.csv
        Columns: city, url, depth, text, word_count
    Outputs : 
    - data/processed/corpus_json.json
        JSON version of the corpus 
    - data/processed/corpus_analysis.xlsx
         sheets with page counts and text-length statistics.
"""
import pandas as pd
import os
from Utils import csv_to_json


BASE_PATH = "data/processed"
CSV_PATH = os.path.join(BASE_PATH, "corpus.csv")
JSON_PATH = os.path.join(BASE_PATH, "corpus_json.json")
EXCEL_PATH = os.path.join(BASE_PATH, "corpus_analysis.xlsx")
 

def main():
    # CSV -> JSON
    df = csv_to_json(CSV_PATH, JSON_PATH)
    
    # Basic counts
    table_total_docs = pd.DataFrame({"Indicator": ["Total number of documents"], "Value": [len(df)]})
    table_pages_city = df.groupby("city").size().reset_index(name="number_of_pages").sort_values(by="number_of_pages", ascending=False)
    table_pages_city_depth = df.groupby(["city", "depth"]).size().reset_index(name="number_of_pages")
    table_provenance = df[["city", "url", "depth"]]

    # Text length statistics
    df['word_count'] = df['text'].str.split().apply(len)
    table_text_stats = df.groupby("city")["word_count"].agg(
        total_pages="count",
        average_words="mean",
        min_words="min",
        max_words="max"
    ).reset_index().round(0)

    table_text_length = df[['city', 'url', 'word_count']]

    total_words_city = df.groupby("city")['word_count'].sum().reset_index(name='total_words')
    pages_per_city = df.groupby("city").size().reset_index(name='num_pages')
    table_city_distribution = pd.merge(total_words_city, pages_per_city, on='city')

    #Export all tables to one Excel file
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
        table_total_docs.to_excel(writer, sheet_name='Total Documents', index=False)
        table_pages_city.to_excel(writer, sheet_name='Pages per City', index=False)
        table_pages_city_depth.to_excel(writer, sheet_name='Pages City & Depth', index=False)
        table_provenance.to_excel(writer, sheet_name='Provenance', index=False)
        table_text_stats.to_excel(writer, sheet_name='Text Stats', index=False)
        table_text_length.to_excel(writer, sheet_name='Text Length per Page', index=False)
        table_city_distribution.to_excel(writer, sheet_name='City Distribution', index=False)

    print(f"All analysis tables exported to Excel: {EXCEL_PATH}")

if __name__ == "__main__":
    main()
