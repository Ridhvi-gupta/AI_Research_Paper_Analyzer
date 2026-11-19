import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DOMAIN_LEXICONS = {
    "🧬 Bioinformatics & Life Sciences": [
        "gene", "protein", "genome", "genomic", "dna", "rna", "molecular", "cell", "cellular",
        "crispr", "sequencing", "biological", "biomarker", "pathway", "mutation", "assay", "tissue"
    ],
    "🤖 Artificial Intelligence & CS": [
        "algorithm", "neural", "deep learning", "transformer", "optimization", "dataset", "accuracy",
        "architecture", "reinforcement", "convolutional", "latency", "inference", "benchmark", "nlp"
    ],
    "🩺 Medicine & Clinical Research": [
        "clinical", "patient", "disease", "treatment", "therapy", "hospital", "diagnosis", "pathology",
        "cohort", "symptom", "efficacy", "trial", "radiology", "lesion", "oncology", "dosage"
    ],
    "⚛️ Physics & Quantum Sciences": [
        "quantum", "thermodynamic", "particle", "energy", "velocity", "magnetic", "optical",
        "frequency", "spectroscopy", "gravitational", "electromagnetic", "simulation", "spin"
    ],
    "🧪 Chemistry & Material Science": [
        "chemical", "synthesis", "catalyst", "polymer", "nanoparticle", "solvent", "reaction",
        "crystallography", "spectrometry", "ligand", "molecule", "reagent", "spectroscopic"
    ]
}

def detect_document_domains(text: str) -> tuple[str, list[str]]:
    text_lower = text.lower()
    scores = {}
    for dom, terms in DOMAIN_LEXICONS.items():
        matches = sum(1 for term in terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower))
        if matches > 0:
            scores[dom] = matches
            
    if not scores:
        return "🌐 Multidisciplinary Science", []
        
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_domains[0][0]
    overlaps = [d[0] for d in sorted_domains[1:3] if d[1] >= max(2, int(sorted_domains[0][1] * 0.4))]
            
    return primary, overlaps

def perform_universal_clustering(papers_dict: dict, k: int = 2) -> dict:
    paper_names = list(papers_dict.keys())
    if not paper_names:
        return {"error": "No documents provided for clustering."}

    corpus = [papers_dict[p]["text"] for p in paper_names]
    n_docs = len(paper_names)
    k = max(1, min(k, n_docs))

    # Guard against empty corpus or stop-words only
    has_words = any(len(re.findall(r'\b[a-zA-Z]{2,}\b', text)) > 0 for text in corpus)
    if not has_words:
        return {"error": "The uploaded documents contain insufficient textual content for semantic clustering."}

    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=800,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        X = vec.fit_transform(corpus)
        feature_names = np.array(vec.get_feature_names_out())
    except ValueError as ve:
        return {"error": f"Clustering TF-IDF error: {str(ve)}. Please upload documents with readable English vocabulary."}

    # Supports 1 paper up to hundreds of papers
    if n_docs == 1:
        coords = np.array([[0.0, 0.0]])
        labels = np.array([0])
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        n_comp = min(2, n_docs)
        pca = PCA(n_components=n_comp, random_state=42)
        trans = pca.fit_transform(X.toarray())
        if n_comp == 1:
            coords = np.column_stack([trans, np.zeros(n_docs)])
        else:
            coords = trans

    cluster_records = []
    for idx, name in enumerate(paper_names):
        text = papers_dict[name]["text"]
        primary_dom, overlaps = detect_document_domains(text)
        domain_badge = primary_dom
        if overlaps:
            domain_badge += f" + {' & '.join(overlaps)}"

        cluster_records.append({
            "Paper": name,
            "Cluster": f"Cluster {labels[idx] + 1}",
            "Cluster_ID": labels[idx],
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "Domain": domain_badge
        })

    df_clusters = pd.DataFrame(cluster_records)

    cluster_topics = {}
    for c_id in range(k):
        sub_indices = np.where(labels == c_id)[0]
        if len(sub_indices) > 0:
            cluster_center = np.mean(X[sub_indices].toarray(), axis=0)
            top_kw_idx = cluster_center.argsort()[::-1][:5]
            cluster_topics[f"Cluster {c_id + 1}"] = list(feature_names[top_kw_idx])

    return {
        "df_clusters": df_clusters,
        "cluster_topics": cluster_topics,
        "k": k
    }