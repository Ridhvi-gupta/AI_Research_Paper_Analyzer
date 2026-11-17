import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Broad Scientific Domain Ontologies
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
        "compound", "crystalline", "spectrometry", "reagent", "spectroscopic", "substrate"
    ]
}

def detect_document_domains(text: str) -> tuple[str, list[str]]:
    """
    Detects primary scientific domain and secondary interdisciplinary overlaps
    (e.g., Computational Biology, AI-driven Medical Imaging).
    """
    text_low = text.lower()
    domain_scores = {}
    
    for domain, terms in DOMAIN_LEXICONS.items():
        score = sum(text_low.count(term) for term in terms)
        domain_scores[domain] = score

    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_domains[0][0] if sorted_domains[0][1] > 2 else "🌐 Multidisciplinary Science"
    
    # Secondary interdisciplinary overlaps
    overlaps = []
    for dom, sc in sorted_domains[1:]:
        if sc >= 3 and sc >= (sorted_domains[0][1] * 0.35):
            overlaps.append(dom.split()[1]) # Short name
            
    return primary, overlaps

def perform_universal_clustering(papers_dict: dict, k: int = 2) -> dict:
    """
    Universal Clustering Engine:
    Performs multi-domain TF-IDF extraction with unigrams + bigrams,
    fits K-Means, computes 2D PCA spatial coords, and assigns domain tags.
    """
    paper_names = list(papers_dict.keys())
    corpus = [papers_dict[p]["text"] for p in paper_names]

    if len(paper_names) < 2:
        return {"error": "Need at least 2 documents for clustering."}

    k = min(k, len(paper_names))

    # Multi-domain TF-IDF with bigrams (e.g., "gene expression", "neural network")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=600,
        sublinear_tf=True
    )
    X = vec.fit_transform(corpus)

    # K-Means Partitioning
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 2D PCA Spatial Coordinates
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())

    # Build per-paper metadata with domain badges
    cluster_records = []
    feature_names = np.array(vec.get_feature_names_out())

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
            "x": coords[idx, 0],
            "y": coords[idx, 1],
            "Domain": domain_badge
        })

    df_clusters = pd.DataFrame(cluster_records)

    # Extract top keywords per cluster
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