import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf(documents: list[str], max_features: int = 1000, ngram_range: tuple = (1, 2)):
    """
    Constructs an optimized TF-IDF vectorizer and feature matrix.
    Safe against short text and empty vocabulary.
    """
    clean_docs = [d if d.strip() else "document literature research" for d in documents]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        token_pattern=r'(?u)\b[A-Za-z]{3,}\b'
    )
    matrix = vectorizer.fit_transform(clean_docs)
    return vectorizer, matrix

def get_top_terms(vectorizer, matrix, top_n: int = 20) -> list[dict]:
    """Extracts top N most informative terms across the corpus ranked by aggregate TF-IDF weight."""
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked_indices = scores.argsort()[::-1][:top_n]
    
    return [
        {
            "term": terms[i],
            "score": round(float(scores[i]), 3),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked_indices)
    ]

def get_document_keywords(vectorizer, matrix, doc_index: int, top_n: int = 10) -> list[dict]:
    """Extracts key distinctive terms for a specific individual document."""
    row = matrix.getrow(doc_index).toarray().flatten()
    terms = vectorizer.get_feature_names_out()
    ranked = row.argsort()[::-1][:top_n]
    return [{"term": terms[i], "score": round(float(row[i]), 3)} for i in ranked if row[i] > 0]
