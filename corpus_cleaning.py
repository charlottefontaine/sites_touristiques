from Utils import *
from parameters import *


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

df_freq_terms.to_csv('data/processed/df_freq_terms.csv', index=False)

# Sum occurrences of each word across all documents
word_frequencies = df_freq_terms.sum(axis=0)

# Sort by descending frequency and get top 20
top_20_words_high = word_frequencies.sort_values(ascending=False).head(20)
print("Top 20 most frequent words:\n", top_20_words_high)

# Sort by ascending frequency and get bottom 50
top_20_words_low = word_frequencies.sort_values(ascending=True).head(50)
print("Top 50 least frequent words:\n", top_20_words_low)

def get_short_terms(term_document_matrix, max_len=3):
    """
    term_document_matrix : pd.DataFrame
        Term–document matrix
    max_len : int
        Maximum term length (default: 2)
    Output:
        List of terms with length <= max_len
    """
    return [
        term for term in term_document_matrix.columns
        if len(term) <= max_len
    ]

short_terms = get_short_terms(df_freq_terms, max_len=3)
print(f"Short terms (length <= 3): {short_terms}")  
print(f"Number of short terms: {len(short_terms)}")


