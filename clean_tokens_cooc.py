#import
import re
import json
import warnings

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    warnings.warn("langdetect not installed. Multi-language filtering disabled. Install with: pip install langdetect")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# NOISE LISTS (CSS, HTML, MULTILINGUAL)

TECHNICAL_NOISE = {
    # CSS/HTML properties
    'padding', 'margin', 'paddingbottom', 'paddingtop', 'marginbottom', 'margintop',
    'background', 'backgroundcolor', 'bgcolor', 'width', 'height', 'maxwidth', 'minwidth',
    'color', 'font', 'fontsize', 'fontweight', 'border', 'borderradius', 'display', 'flex',
    'position', 'absolute', 'relative', 'hidden', 'visible', 'opacity', 'zindex',
    'cursor', 'pointer', 'px', 'em', 'rem', 'rgba', 'hex', 'url', 'href', 'src',
    'div', 'span', 'class', 'id', 'style', 'script', 'input', 'button', 'form',
    'click', 'hover', 'submit', 'loading', 'search', 'menu', 'nav', 'header', 'footer'
}

MULTILINGUAL_NOISE = {
    # Common words in DE, NL, ES, IT, FR to filter mixed corpus
    'und', 'der', 'die', 'das', 'mit', 'für', 'von', 'ist', 'nicht', 
    'aan', 'het', 'een', 'van', 'en', 'op', 'te', 'zijn', 'voor',   
    'el', 'la', 'de', 'que', 'y', 'en', 'un', 'por', 'con', 'para', 
    'il', 'di', 'che', 'la', 'un', 'per', 'non', 'una', 'con',      
    'le', 'la', 'les', 'de', 'et', 'un', 'une', 'est', 'pour',        
    'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'al', 'en', 
    'con', 'por', 'para', 'sin', 'sus', 'su', 'mi', 'tu', 'te', 'se', 'lo', 
    'me', 'nos', 'es', 'son', 'fue', 'era', 'muy', 'ms', 'pero', 'si', 'no',
    'todo', 'cada', 'est', 'hay', 'vuelo', 'aeropuerto', 'servicio', 'traslado',
    'os', 'as', 'um', 'uma', 'ao', 'aos', 'do', 'dos', 'da', 'das', 'no', 'nas',
    'pela', 'pelo', 'com', 'sem', 'que', 'se', 'foi', 'so', 'bom', 'bem',
    'teixeira', 'aldeia', 'mitos', 'lendas', 'livres', 'edifcio',
    'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'du', 'au', 'aux',
    'en', 'dans', 'par', 'pour', 'sur', 'avec', 'sans', 'ce', 'cet', 'cette',
    'ces', 'qui', 'que', 'quoi', 'est', 'sont', 'il', 'ils', 'elle', 'elles',
    'nous', 'vous', 'votre', 'nos', 'vos', 'leur', 'leurs', 
    'utiliss', 'politique', 'cookies', 'publicits', 'fournisseur',
    'jours', 'nom', 'pa', 'catgoriss', 'catgories', 'dhte', 'hote', 'ncessaires', 
    'necessaires', 'inconnu', 'peut', 'peuvent', 'amliorer', 'ameliorer', 'nont', 
    'catgorie', 'categorie', 'paramtres', 'parametres', 'notre', 'votre', 'refuser', 
    'accepter', 'rapportant', 'cidessous', 'ci-dessous', 'collectant', 'jour', 'llc',
    'sont', 'ont', 'les', 'des', 'aux', 'du', 'par', 'pour', 'sur', 'dans', 
    'grup', 'divisi', 'iglesia', 'jove', 'obispo', 'casas', 'grau',
    'roze', 'blauw', 'ggd', 'meldpunt', 'nederland', 'discriminatie', 'netwerk', 'bij'
}

PROJECT_STOPWORDS_cooc = {
    # cookies
    'cookie', 'cookies', 'policy', 'privacy', 'rights', 'reserved', 
    'copyright', 'consent', 'content', 'click', 'website', 'page', 
    'home', 'main', 'skip', 'loading', 'browser', 'newsletter', 
    'subscribe', 'email', 'contact', 'address', 'tel', 'fax',
    'facebook', 'twitter', 'instagram', 'youtube', 'blog','cookiehub', 'doubleclicknet', 'fbp', 'clsk', 'clck', 
    'analytiques', 'dsactives', 'autoriser', 'autorises',
    'consulter', 'confidentialit', 'dclaration', 'dentre',
    'abonnezvous', 'abonnieren', 'accesibilidad', 'privacy',
    'settings', 'preferences',

    # meaningless verbs
    'visit', 'visiting', 'visitor', 'visitors', 
    'located', 'situated', 'location', 
    'experience', 'discover', 'explore', 'enjoy', 
    'offer', 'offers', 'offering', 'provide', 'provides', 'including',
    'make', 'take', 'get', 'find', 'look', 'see', 'view',
    'start', 'end', 'open', 'close', 'check', 'book',
    
    # selling adjectives 
    'best', 'great', 'good', 'nice', 'beautiful', 'amazing', 
    'wonderful', 'perfect', 'famous', 'unique', 
    'top', 'high', 'large', 'small', 'new', 'old',
    'special', 'available', 'accessible', 'additional',
    
    # time words
    'time', 'times', 'day', 'days', 'today', 'daily',
    'hour', 'hours', 'minute', 'minutes', 'year', 'years',
    'week', 'weekend', 'month', 'months',
    'price', 'ticket', 'entry', 'entrance', 
    'people', 'person', 'group', 'family', 'child', 'children', 'adult',
    'number', 'amount', 'total', 'level',
    
    # common tourism words
    'according', 'action', 'active', 'activities',
    'advice', 'advance', 'advantage', 'allow', 'allows', 'allowed',
    'also', 'always', 'already', 'almost', 'another', 'around',
    'area', 'city', 'town', 'centre', 'place', 'places',
    'part', 'way', 'well', 'need', 'must', 'tour', 'travel',
    'relaxing', 'wealth', 'incredible', 'souvenir'

}



# cleaning functions

def is_technical_garbage(token: str) -> bool:
    """Detects CSS/HTML artifacts"""
    if token in TECHNICAL_NOISE: return True
    if re.search(r'(px|em|rem|rgba|http|www)', token): return True
    if re.search(r'[0-9]', token): return True 
    if len(token) > 25: return True 
    return False

def advanced_clean_tokens(tokens, project_stopwords=None, min_length=3):
    """
    Main cleaning pipeline
    """
    if project_stopwords is None:
        project_stopwords = PROJECT_STOPWORDS_cooc
        
    
    
    full_stoplist = stop_words | project_stopwords | MULTILINGUAL_NOISE
    
    cleaned = []
    
    for t in tokens:
        t = t.lower()
        t = re.sub(r"[^a-z]", "", t) 
        
        
        if not t or len(t) < min_length:
            continue

        if re.search(r'\d', t):
            continue

        
        if t in full_stoplist:
            continue
            
        
        if is_technical_garbage(t):
            continue
            
        
        t = lemmatizer.lemmatize(t)
        
        cleaned.append(t)
        
    return cleaned

def load_and_clean_json(json_path, min_doc_length=10):
    """
    Loads JSON and applies cleaning + language filtering
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cleaned_sequences = []
    
    for doc in data:
        text = doc.get("text", "")
        
        
        if HAS_LANGDETECT and len(text) > 50:
            try:
                if detect(text) != 'en':
                    continue 
            except:
                continue
                
        tokens = text.split()
        
        
        cleaned = advanced_clean_tokens(tokens)
        
       
        if len(cleaned) >= min_doc_length:
            cleaned_sequences.append(cleaned)
            
    print(f"Loaded and cleaned {len(cleaned_sequences)} documents.")
    return cleaned_sequences

