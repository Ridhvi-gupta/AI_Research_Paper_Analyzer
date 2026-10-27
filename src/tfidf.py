from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf(documents: list[str], max_features: int = 1000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix

def get_top_terms(vectorizer, matrix, top_n: int = 15) -> list[dict]:
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = scores.argsort()[::-1][:top_n]
    return [{"term": terms[i], "score": round(float(scores[i]), 3)} for i in ranked]