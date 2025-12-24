from Utils import *
from parameters import *
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer 

#---------------------------------
# Part1 . Dimensionality reduction
#---------------------------------

# dictionary creation
dict_tokens = tokenize_json_by_city_url("data/processed/corpus_json.json")

# report
cities = set(key[0] for key in dict_tokens.keys())
print(f"Cities present: {sorted(cities)}")
print(f"Total entries: {len(dict_tokens)}")

# term-document matrix
df_tdm = build_term_document_matrix(dict_tokens)
print(f"Term-document matrix shape: {df_tdm.shape}")
print("Preview of term-document matrix:")
print(df_tdm.head())

# first cleaning
df_tdm_cleaned = clean_term_document_matrix(df_tdm, None, keep_numbers=False)
print(f"Term-document matrix shape after cleaning: {df_tdm_cleaned.shape}")
print("Preview of cleaned lowercased term-document matrix:")
print(df_tdm_cleaned.head())

#social media normalized
df_tdm_cleaned = normalize_social_media_terms(df_tdm_cleaned)
print(f"Term-document matrix shape after social media normalization: {df_tdm_cleaned.shape}")
print("Preview after social media normalization:")  
print(df_tdm_cleaned.head())

# remove standard stopwords
df_tdm_no_stopwords = remove_nltk_stopwords(df_tdm_cleaned)
print(f"Term-document matrix shape after NLTK stopwords removal: {df_tdm_no_stopwords.shape}")    
print("Preview after removing NLTK stopwords:")
print(df_tdm_no_stopwords.head())

# lemmatization
df_tdm_lemmatized = lemmatize_matrix_nltk(df_tdm_no_stopwords)
print(f"Term-document matrix shape after lemmatization: {df_tdm_lemmatized.shape}")
print("Preview of lemmatized term-document matrix:")
print(df_tdm_lemmatized.head())

# remove project-specific stopwords
df_tdm_project_stopwords_removed = remove_project_stopwords(df_tdm_lemmatized, project_stopwords)
print(f"Term-document matrix shape after removing project-specific stopwords: {df_tdm_project_stopwords_removed.shape}")    
print("Preview after removing project-specific stopwords:")   
print(df_tdm_project_stopwords_removed.head())

#remove miniwords
df_tdm_miniwords_removed = remove_miniwords(df_tdm_project_stopwords_removed, min_length=3)
print(f"Term-document matrix shape after removing miniwords: {df_tdm_miniwords_removed.shape}")    
print("Preview after removing miniwords:")      
print(df_tdm_miniwords_removed.head())


# frequency analysis
df_freq_terms = filter_terms_by_frequency(df_tdm_miniwords_removed)


print("Preview of term frequencies:")    
print(df_freq_terms.head(10))
print(f"Term-document matrix shape after frequency filtering: {df_freq_terms.shape}")
print("Preview of term-document matrix after frequency filtering:")
print(df_freq_terms.head())


#----------------------
# Part2 . Exploration 
#----------------------

# Sum occurrences of each word across all documents
word_frequencies = df_freq_terms.sum(axis=0)

short_terms = get_short_terms(df_freq_terms, max_len=3)
print(f"Short terms (length <= 3): {short_terms}")  
print(f"Number of short terms: {len(short_terms)}")

#-------------------------------
# Part3 . Csv file modification 
#-------------------------------

# avoid SettingWithCopyWarning and ensure we have an independent DataFrame 
df_freq_terms = df_freq_terms.copy() 

# Extract the city from the index 
df_freq_terms["city"] = [t[0] for t in df_freq_terms.index] 

# # Reorganise so that city is the first column 
cols = ["city"] + [c for c in df_freq_terms.columns if c != "city"] 
df_freq_terms = df_freq_terms[cols] 

df_freq_terms.to_csv('data/processed/df_freq_terms.csv', index=False) 
print(f"Filtered term-document matrix saved: {df_freq_terms.shape}") 

# Diagnostics: compute word frequencies only on numeric term columns 
terms_only = [c for c in df_freq_terms.columns if c != "city"] 
if len(terms_only) == 0: 
    print("Warning: no term columns found in df_freq_terms (only 'city' present?)") 
else: 
    word_frequencies = df_freq_terms[terms_only].sum(axis=0) 
    print("Top 20 most frequent words:\n", word_frequencies.sort_values(ascending=False).head(20)) 
    print("Bottom 20 least frequent words:\n", word_frequencies.sort_values(ascending=True).head(20)) 



#-------------------------------
# Part4 . TF-IDF matrix
#-------------------------------

terms_only = [c for c in df_freq_terms.columns if c != "city"] 
tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True) 
X_tfidf = tfidf_transformer.fit_transform(df_freq_terms[terms_only].values) 
df_tfidf = pd.DataFrame( 
    X_tfidf.toarray(), 
    columns=terms_only, 
    index=df_freq_terms.index 

) 

df_tfidf["city"] = df_freq_terms["city"].values 

print(f"TF-IDF matrix shape: {df_tfidf.shape}") 
print(df_tfidf.head()) 

# Aggregation by city and standardisation 

tfidf_by_city = df_tfidf.groupby("city").mean() 
tfidf_by_city_norm = tfidf_by_city.div(tfidf_by_city.sum(axis=1), axis=0) 

print("TF-IDF normalized by city:") 
print(tfidf_by_city_norm.head()) 

# Verification 
row_sums = tfidf_by_city_norm.sum(axis=1) 
print("Sum of normalized TF-IDF per city:", row_sums) 

# Save TF-IDF matrices
df_tfidf.to_csv("data/processed/df_tfidf_documents.csv",index=False)
print("Saved: data/processed/df_tfidf_documents.csv")

tfidf_by_city_norm.to_csv("data/processed/df_tfidf_by_city.csv")
print("Saved: data/processed/df_tfidf_by_city.csv")


 

 