# 🔬 ResearchLens AI — Universal Academic Paper Analyzer & Adaptive Research Gap Mining Engine

**Domain**: Universal / Open-Domain Literature Intelligence  
*(Spanning Artificial Intelligence, Deep Learning, Natural Language Processing, Biomedical Sciences, Clinical Medicine, Physical & Chemical Sciences, Engineering, Economics, Social Sciences, and Cross-Disciplinary Research)*  

**Authors & Team Members**:
- **Ridhvi Gupta** (S25CSEU548)
- **Shaurya Pathak** (S25CSEU0513)

---

## 📌 Executive Summary

**ResearchLens AI** is an autonomous, open-domain research intelligence system engineered to ingest, synthesize, cluster, and mine uncharted research directions across **any scientific or academic field**. 

Unlike conventional citation trackers or static keyword search tools, ResearchLens AI couples **unsupervised cross-domain topic modeling** with an **adaptive, self-learning hybrid gap mining engine** (1D Convolutional Neural Network sequence classifier + continuous semantic vocabulary induction). 

Researchers can review, validate, or reject discovered limitations in real-time, generate an executive **A4 Research Dossier**, and download **amended revisions of their papers** containing an auditable *"Validated Research Gaps & Strategic Amendments"* section in any format (`PDF`, `DOCX`, `Markdown`, `Plain Text`, `JSON`, or `HTML`).

---

## 🌟 Core Highlights & Capabilities

### 1. 🌐 Universal Multi-Format Document Ingestion
Ingest literature in any raw digital format without manual copy-pasting or file conversions:
- **Adobe PDF (`.pdf`)**: Native stream parsing via PyMuPDF with page count and font rendering.
- **Microsoft Word (`.docx`)**: Zero-dependency OpenXML element-tree parser supporting tables, text boxes, and varied XML namespaces.
- **Plain Text & Markdown (`.txt`, `.md`)**: UTF-8 compliant academic notes and drafts.
- **Structured Data (`.csv`, `.tsv`, `.json`)**: Tabular paper repositories, arXiv API dumps, or Semantic Scholar corpus exports.
- **Web Publications (`.html`, `.htm`)**: Tag-stripped, section-aware ingestion of online preprint articles.
- **Smart Reference Filtering**: Section-aware regex stripping automatically excises bibliographies and citation lists to focus analysis purely on original claims.

### 2. 🌍 Universal Domain-Agnostic Intelligence (Any Field of Study)
The system is designed from the ground up to analyze **any field or discipline**:
- Computer Science, Machine Learning & AI
- Genetics, Molecular Biology & Life Sciences
- Clinical Medicine, Pharmacology & Public Health
- Physics, Quantum Computing & Materials Science
- Chemistry, Nanotechnology & Chemical Engineering
- Economics, Social Sciences, Environmental Science, Humanities & Beyond

### 3. 📈 True Elastic Scalability ($1 \le N < \infty$)
Whether analyzing a single novel preprint ($N = 1$) or a corpus of hundreds of papers ($N \gg 100$):
- **Single Paper Mode**: Generates self-contained topological maps, vocabulary histograms, and deep gap attribution without index out-of-range or dimension collapse errors.
- **Multi-Paper Mode**: Dynamically tunes K-Means clustering ($2 \le k \le 15$), pairwise cosine similarity heatmaps, and 2D Principal Component Analysis (PCA) projections.

### 4. 🧬 Adaptive Self-Learning Gap Mining Engine
Rather than relying solely on hardcoded static dictionaries:
- **Continuous Vocabulary Induction**: Automatically scans contrastive syntactic constructs (e.g., *"remains intractable"*, *"paucity of"*, *"degrades under"*, *"fails to generalize"*) and dynamically inducts new trigger words into `data/learned_gap_triggers.json`.
- **1D-CNN Sequence Modeling**: A trained 1D Convolutional Neural Network evaluates grammatical and contextual transitions in sentence embeddings, assigning high probability ($P_{\text{CNN}} \in [0, 1]$) to authentic scientific limitation claims.
- **Hybrid Confidence Scoring**:
  $$\text{Score}_{\text{Hybrid}} = \alpha \cdot \text{Score}_{\text{Rules}} + \beta \cdot P_{\text{CNN}}$$
- **Confidence Sensitivity Slider**: Interactively tune sensitivity ($0.20 \le \tau \le 0.85$) to filter between broad exploratory leads and strictly high-confidence bottlenecks.

### 5. ✍️ Researcher Gap Validation & Multi-Format Amended Paper Export
A closed-loop human-in-the-loop validation interface:
- **One-Click Gap Actions**: Researchers can click **✓ Validate Gap** or **✕ Reject Gap**.
- **Automated Strategic Resolutions**: The platform infers the deficit classification (Computational Complexity, Data Scarcity, Factuality/Hallucination, Architectural Bounds, or Empirical Generalization) and formulates an actionable research solution.
- **Multi-Format Export**: Download the enriched document containing the original text plus the formal **"Section: Validated Research Gaps & Methodological Amendments"** in:
  1. `PDF (.pdf)` — Formatted letter document with stylized callout boxes.
  2. `Word Document (.docx)` — Native Word OpenXML document with headers and color-coded amendment tables.
  3. `Markdown (.md)` — GitHub-flavored markdown with blockquotes and headers.
  4. `Plain Text (.txt)` — Standard plain text document.
  5. `JSON (.json)` — Machine-readable structured payload containing original text, timestamps, and validated gap metadata.
  6. `HTML (.html)` — Dark-mode responsive web report with CSS cards.

### 6. 📊 Executive Research Dossier (ReportLab PDF)
Compiles a publication-ready multi-page A4 PDF report detailing cross-domain metadata, token statistics, and an executive breakdown of all verified research directions.

---

## 🏛️ System Architecture

```text
📁 AI_Research_Paper_Analyzer/
├── 📄 app.py                     # Streamlit frontend & reactive routing engine
├── 📁 src/
│   ├── 📄 universal_extractor.py # Ingestion engine (PDF, DOCX, TXT, MD, CSV, JSON, HTML)
│   ├── 📄 preprocessing.py       # Sentence segmentation, cleaning & tokenization
│   ├── 📄 clustering.py          # Domain categorization, K-Means & 2D PCA spatial maps
│   ├── 📄 gap_rules.py           # Adaptive lexical rules & persistent lexicon manager
│   ├── 📄 dynamic_gap_learner.py # Word-pattern induction & vocabulary induction
│   ├── 📄 cnn_model.py           # 1D-CNN neural classifier & batch vector inference
│   ├── 📄 train_cnn.py           # Pipeline to train/retrain the Keras 1D-CNN model
│   ├── 📄 document_exporter.py   # Multi-format amended paper generator (6 formats)
│   ├── 📄 report_generator.py    # Executive A4 PDF Dossier compiler (ReportLab)
│   ├── 📄 eda.py                 # Exploratory data analysis & vocabulary frequencies
│   ├── 📄 tfidf.py               # TF-IDF term-weighting utility
│   └── 📄 word2vec.py            # Semantic embedding extraction
├── 📁 models/
│   ├── 📄 cnn_gap_model.keras    # Pre-trained 1D Convolutional Neural Network
│   └── 📄 tokenizer.pkl          # Pickled Keras sequence tokenizer
├── 📁 data/
│   ├── 📄 history_log.json       # Audit log of uploaded literature
│   ├── 📄 learned_gap_triggers.json # Persistent self-learned gap trigger lexicon
│   └── 📁 dataset/
│       └── 📄 gap_dataset.csv    # Benchmark training corpus for 1D-CNN
└── 📄 README.md                  # Project documentation
```

---

## 🛠️ Step-by-Step Installation & Setup

### Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **OS**: Windows, macOS, or Linux

### 1. Open Terminal in Project Directory
Open your terminal or VS Code PowerShell in the root directory:
```powershell
cd c:\Users\lenovo\Documents\AI_Research_Paper_Analyzer
```

### 2. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

---

## 🚀 Running the Platform

### Launch the Streamlit Web Dashboard:
```powershell
streamlit run app.py
```

Once started, open your web browser at:
```text
http://localhost:8501
```

---

## 🧠 (Optional) Retraining the 1D-CNN Sequence Classifier

If you expand the training dataset in `data/dataset/gap_dataset.csv`, you can retrain the model with:

```powershell
python src/train_cnn.py
```

This compiles a 1D Convolutional Neural Network with an embedding layer, 1D convolution, global max pooling, dense ReLU projection, and a sigmoid output unit, saving the updated model to `models/cnn_gap_model.keras` and the tokenizer to `models/tokenizer.pkl`.

---

## 📖 Walkthrough: Using the Application

1. **🏠 Home**: Overview of the pipeline with one-click access to load the multi-domain benchmark corpus or upload custom papers.
2. **🚀 Upload & Analyze Hub**: Drop any combination of files (`PDF`, `DOCX`, `TXT`, `MD`, `CSV`, `JSON`, `HTML`). The system will parse them, classify their disciplines, generate a semantic similarity heatmap, and surface candidate research gaps.
3. **📄 Multi-Format Papers Library**: Review loaded papers, check page/word counts, unload individual documents from memory (`❌`), or delete records from the persistent audit log (`🗑️`).
4. **📈 Text Analysis & EDA**: View cross-corpus vocabulary distributions and token frequencies.
5. **🔗 Cosine Similarity**: Pairwise cross-paper similarity matrices and top similar paper pairs.
6. **🧩 Interdisciplinary Clustering**: 2D PCA spatial maps partitioning literature into thematic clusters with discovered keywords.
7. **🔍 Adaptive Gap Mining & Suggestions**: Filter candidate gaps using the interactive sensitivity slider, validate genuine limitations (**✓ Validate Gap**), and download amended papers with applied strategic proposals in your choice of 6 formats.
8. **🧠 Learned Vocabulary & AI Rules**: Inspect the self-learned trigger lexicon or teach the model new domain-specific deficit words.
9. **📑 Dossier Reports**: Export a multi-page executive A4 PDF dossier or download individual modified papers.
10. **❓ FAQ & Research Q&A**: Ask natural language questions across all ingested literature.

---

## 📜 Academic Attribution & License

Developed for advanced academic literature intelligence and autonomous research gap discovery.  
Released under the **MIT License**.