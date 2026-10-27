from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(matrix):
    return cosine_similarity(matrix)

def get_similarity_pairs(similarity_matrix, paper_names: list[str]) -> list[dict]:
    pairs = [
        {"Paper 1": paper_names[i], "Paper 2": paper_names[j], "similarity": round(float(similarity_matrix[i][j]), 3)}
        for i in range(len(paper_names)) for j in range(i + 1, len(paper_names))
    ]
    return sorted(pairs, key=lambda x: x["similarity"], reverse=True)