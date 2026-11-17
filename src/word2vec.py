from gensim.models import Word2Vec
import re

def train_word2vec_model(sentences: list[str], vector_size: int = 100, window: int = 5, min_count: int = 1) -> Word2Vec:
    """Trains a continuous Skip-Gram Word2Vec embedding model over extracted sentence tokens."""
    tokenized_corpus = []
    for s in sentences:
        tokens = re.findall(r'\b[A-Za-z]{3,}\b', s.lower())
        if tokens:
            tokenized_corpus.append(tokens)

    if not tokenized_corpus:
        tokenized_corpus = [["research", "paper", "analysis", "limitation", "method"]]

    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=2,
        epochs=15
    )
    return model

def expand_keywords_with_word2vec(base_keywords: list[str], w2v_model: Word2Vec, topn: int = 2) -> list[str]:
    """
    Dynamically discovers domain-specific synonyms (e.g. 'limitation' -> 'bottleneck', 'drawback')
    to expand research gap detection.
    """
    expanded = set(base_keywords)
    if w2v_model is None or not hasattr(w2v_model, "wv"):
        return list(expanded)

    for kw in base_keywords:
        kw_clean = kw.lower().strip()
        if kw_clean in w2v_model.wv:
            try:
                sims = w2v_model.wv.most_similar(kw_clean, topn=topn)
                for word, similarity in sims:
                    if similarity > 0.45 and len(word) > 3:
                        expanded.add(word)
            except Exception:
                continue

    return list(expanded)

def get_similar_words(model: Word2Vec, word: str, topn: int = 5) -> list[tuple[str, float]]:
    """Returns top N most similar terms for a given keyword."""
    if model and hasattr(model, "wv") and word.lower() in model.wv:
        return model.wv.most_similar(word.lower(), topn=topn)
    return []