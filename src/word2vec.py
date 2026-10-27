import numpy as np
from gensim.models import Word2Vec

def train_word2vec(tokenized_documents: list[list[str]], vector_size: int = 100, window: int = 5):
    docs = [doc for doc in tokenized_documents if doc] or [["deep", "learning", "research"]]
    return Word2Vec(sentences=docs, vector_size=vector_size, window=window, min_count=1, workers=2, seed=42)

def document_vector(model: Word2Vec, tokens: list[str]) -> np.ndarray:
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)