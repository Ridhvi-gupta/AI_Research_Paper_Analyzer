import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Auto-download required NLTK tokenizers and lexicons
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "on", "as", "are", "by", "an", "be", "we", "from", "which", "our"}

def clean_academic_text(text: str) -> str:
    """
    Cleans raw academic PDF text by stripping citations, figure/table captions,
    LaTeX equations, and formatting noise before NLP processing.
    """
    if not text:
        return ""
    
    # Remove bracketed numerical citations like [1], [2, 3], [12-15]
    text = re.sub(r'\[[0-9,\s\-–]+\]', '', text)
    
    # Remove author citations like (Devlin et al., 2019) or (Vaswani, 2017)
    text = re.sub(r'\([A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?\)', '', text)
    
    # Remove Figure and Table captions
    text = re.sub(r'(?:Figure|Fig\.|Table)\s+\d+[:\.\-][^\n]+', '', text, flags=re.IGNORECASE)
    
    # Remove URLs and DOIs
    text = re.sub(r'https?://\S+|www\.\S+|doi\.org/\S+', '', text)
    
    # Remove LaTeX math artifacts (e.g. \lambda, \sum_{i=1}^N)
    text = re.sub(r'\\[A-Za-z]+(?:\{[^}]*\})?', '', text)
    
    # Normalize whitespaces, tabs, and hyphens from line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # Rejoin hyphenated words split across lines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_sentences(text: str) -> list[str]:
    """
    Segments raw text into clean, individual grammatical claims.
    Filters out noisy fragments (< 6 words or < 25 characters).
    """
    cleaned = clean_academic_text(text)
    try:
        raw_sents = sent_tokenize(cleaned)
    except Exception:
        raw_sents = [s.strip() for s in cleaned.split('.') if s.strip()]

    valid_sentences = []
    for s in raw_sents:
        s_clean = s.strip()
        words = s_clean.split()
        # Keep valid sentences, reject short headers or math fragments
        if len(words) >= 6 and len(s_clean) >= 25:
            valid_sentences.append(s_clean)
            
    return valid_sentences

def preprocess_text(text: str) -> str:
    """Lowercases, removes stopwords, and lemmatizes text for TF-IDF / vector modeling."""
    cleaned = clean_academic_text(text)
    tokens = re.findall(r'\b[A-Za-z]{3,}\b', cleaned.lower())
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(lemmatized)

def clean_tokens(tokens: list[str]) -> list[str]:
    """Cleans a raw list of tokens."""
    return [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha() and t.lower() not in stop_words and len(t) > 2]