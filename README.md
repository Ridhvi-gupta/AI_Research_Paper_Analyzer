# AI Research Paper Analyzer + Research Gap Mining

**Domain**: Data Analysis + Natural Language Processing + Deep Learning    
**Team Members**: 
- Ridhvi Gupta (S25CSEU548)
- Shaurya Pathak (S25CSEU0513)  
**Faculty Guide**: Mrs. Susmita Das  

---

## 📌 Project Overview
Automated literature review and research gap mining system that extracts, preprocesses, clusters, and analyzes academic papers to propose unexplored areas and candidate research gaps.

### 5-Stage Architecture:
1. **Collect & Extract** (`pdf_extractor.py`, `metadata.py`): Ingests PDF research papers and extracts structured text, page counts, and metadata via PyMuPDF.
2. **Clean & Preprocess** (`preprocessing.py`): Lowercasing, noise reduction, sentence segmentation, stopword filtering, and WordNet lemmatization.
3. **Analyze & Represent** (`eda.py`, `tfidf.py`, `word2vec.py`): Computes keyword frequencies, WordCloud visual representations, TF-IDF feature matrices, and continuous Word2Vec semantic embeddings.
4. **Compare & Group** (`similarity.py`, `clustering.py`): Cross-paper Pairwise Cosine Similarity matrix and K-Means thematic clustering visualized via 2D PCA projection.
5. **Detect Potential Gaps** (`gap_rules.py`, `cnn_model.py`, `train_cnn.py`, `gap_detector.py`): Hybrid scoring engine pairing surface lexical cue matching with a 1D Convolutional Neural Network (CNN) sentence classifier.
6. **Reporting & Dashboard** (`report_generator.py`, `app.py`): Generates downloadable A4 PDF reports via ReportLab and serves an interactive local Streamlit dashboard.

---

## 🚀 Setup & Execution Guide (VS Code)

### Step 1: Activate Virtual Environment
In your VS Code terminal:
```bash
.\venv\Scripts\activate

## Commnad to run in terminal 
# python -m streamlit run app.py

# OR

## If your dependencies are installed inside a venv folder, run:

# .\venv\Scripts\Activate.ps1
# python -m streamlit run app.py

