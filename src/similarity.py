import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(matrix) -> np.ndarray:
    """Computes pairwise cosine similarity matrix from vector representations."""
    return cosine_similarity(matrix)

def get_similarity_pairs(similarity_matrix: np.ndarray, paper_names: list[str], top_n: int = 10) -> list[dict]:
    """
    Extracts sorted pairwise document similarity scores with rank order.
    Excludes diagonal identity pairs (Paper A <-> Paper A).
    """
    pairs = []
    n = len(paper_names)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_matrix[i, j])
            pairs.append({
                "Paper 1": paper_names[i],
                "Paper 2": paper_names[j],
                "similarity": round(score, 3),
                "similarity_pct": f"{score * 100:.1f}%"
            })

    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_n]

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes lexical Jaccard set similarity between two documents."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return round(intersection / union, 4)

def build_similarity_dataframe(similarity_matrix: np.ndarray, paper_names: list[str]) -> pd.DataFrame:
    """Constructs a clean labeled DataFrame for heatmaps and tabular visualization."""
    short_names = [p[:22] for p in paper_names]
    return pd.DataFrame(similarity_matrix, index=short_names, columns=short_names)
