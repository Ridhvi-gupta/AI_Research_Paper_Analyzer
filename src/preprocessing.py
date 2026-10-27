import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

for res in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_and_tokenize(text: str) -> tuple[str, list[str]]:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", re.sub(r"\s+", " ", text.lower()))
    tokens = [LEMMATIZER.lemmatize(w) for w in word_tokenize(cleaned) if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens), tokens

def clean_text(text: str) -> str:
    return clean_and_tokenize(text)[0]

def get_tokens(text: str) -> list[str]:
    return clean_and_tokenize(text)[1]

def extract_sentences(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip().split()) >= 4]