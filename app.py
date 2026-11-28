import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Backend Module Imports
from src.universal_extractor import extract_universal_document
from src.gap_rules import load_learned_triggers, induce_gap_vocabulary_from_corpus, save_learned_triggers
from src.cnn_model import predict_gap_probability, batch_predict_gap_probabilities
from src.report_generator import generate_pdf_report
from src.clustering import perform_universal_clustering, detect_document_domains
from src.document_exporter import export_modified_file, build_modified_document_content

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ResearchLens AI | Research Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. SESSION STATE & PERSISTENT HISTORY INITIALIZATION
# ==============================================================================
HISTORY_FILE = os.path.join("data", "history_log.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(entries):
    os.makedirs("data", exist_ok=True)
    existing = load_history()
    existing_names = {item["filename"] for item in existing}
    for e in entries:
        if e["filename"] not in existing_names:
            existing.insert(0, e)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(existing[:100], f, indent=2)

def delete_from_history(filenames):
    existing = load_history()
    to_delete = set(filenames)
    updated = [item for item in existing if item.get("filename") not in to_delete]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    return updated

def clear_all_history():
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "papers" not in st.session_state:
    st.session_state.papers = {}
if "validated_gaps" not in st.session_state:
    st.session_state.validated_gaps = {}
if "rejected_gaps" not in st.session_state:
    st.session_state.rejected_gaps = set()
if "newly_learned_words" not in st.session_state:
    st.session_state.newly_learned_words = []
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "🏠 Home"

# ==============================================================================
# 3. DYNAMIC THEME STYLING WITH GLOWING PIPELINE BOXES
# ==============================================================================
is_dark = (st.session_state.theme_mode == "Dark")

if is_dark:
    bg_color = "#080c14"
    card_bg = "rgba(14, 19, 32, 0.95)"
    sidebar_bg = "#0b101c"
    text_color = "#f8fafc"
    sub_text = "#cbd5e1"
    border_accent = "#ff7700"
    glow_box = "0 0 18px rgba(255, 136, 0, 0.85), 0 0 32px rgba(255, 170, 0, 0.45), inset 0 0 10px rgba(255, 119, 0, 0.2)"
else:
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    sidebar_bg = "#f1f5f9"
    text_color = "#0f172a"
    sub_text = "#475569"
    border_accent = "#ea580c"
    glow_box = "0 0 16px rgba(234, 88, 12, 0.6), 0 0 25px rgba(245, 158, 11, 0.4)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {text_color};
}}

.stApp {{
    background: radial-gradient(circle at 10% 8%, rgba(255, 119, 0, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 90% 92%, rgba(255, 170, 0, 0.06) 0%, transparent 45%),
                {bg_color};
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 2px solid {border_accent}44 !important;
}}

.pipeline-screenshot-container {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin: 18px 0 24px 0;
    overflow-x: auto;
    padding: 14px 6px;
}}

.pipeline-stage-box {{
    flex: 1;
    min-width: 140px;
    background: #090d16;
    border: 2px solid #ff8800;
    border-radius: 9px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: {glow_box};
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}

.pipeline-stage-box:hover {{
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 25px rgba(255, 170, 0, 0.95), 0 0 45px rgba(255, 102, 0, 0.7);
    border-color: #ffe066;
}}

.stage-num {{
    font-size: 0.72rem;
    font-weight: 800;
    color: #ff9900;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
}}

.stage-title {{
    font-size: 0.88rem;
    font-weight: 800;
    color: #ffffff;
    margin: 3px 0;
}}

.stage-desc {{
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}}

.pipeline-arrow-glow {{
    font-size: 1.4rem;
    font-weight: 900;
    color: #ffaa00;
    padding: 0 4px;
    filter: drop-shadow(0 0 8px #ff7700);
}}

.step-instruction-card {{
    background: {card_bg};
    border-left: 5px solid #ff7700;
    border-top: 1.5px solid rgba(255, 119, 0, 0.3);
    border-right: 1.5px solid rgba(255, 119, 0, 0.3);
    border-bottom: 1.5px solid rgba(255, 119, 0, 0.3);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}}

.step-badge {{
    background: rgba(255, 119, 0, 0.2);
    border: 1.5px solid #ffaa00;
    color: #ffe066;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 2px 10px;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 6px;
}}

.brand-title {{
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ff5500 0%, #ffaa00 35%, #ffe066 50%, #ff5500 70%, #ffaa00 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}
.brand-sub {{
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: {border_accent};
    margin-top: 2px;
    margin-bottom: 16px;
}}

.kpi-card {{
    background: {card_bg};
    border: 2px solid {border_accent};
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: {glow_box};
}}

.gap-card {{
    background: {card_bg};
    border-left: 5px solid {border_accent};
    border-top: 1.5px solid {border_accent}44;
    border-right: 1.5px solid {border_accent}44;
    border-bottom: 1.5px solid {border_accent}44;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}}

.gap-text {{
    font-size: 0.92rem;
    color: {text_color};
    line-height: 1.6;
    background: rgba(255, 119, 0, 0.05);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 10px;
    border: 1px dashed rgba(255, 119, 0, 0.25);
}}

.gap-suggestion-box {{
    background: rgba(16, 185, 129, 0.12);
    border: 1.5px solid #10b981;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
}}

.stButton>button {{
    background: linear-gradient(135deg, #ff9900 0%, #ff5500 50%, #ffaa00 100%) !important;
    background-size: 200% auto !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border: 2px solid #ffe066 !important;
    border-radius: 9px !important;
    padding: 8px 18px !important;
    box-shadow: 0 0 14px rgba(255, 102, 0, 0.6) !important;
}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. TOP HEADER BAR WITH SUN TOGGLE
# ==============================================================================
top_h1, top_h2 = st.columns([5.5, 1.3])
with top_h1:
    st.markdown('<p class="brand-title">ResearchLens AI</p>', unsafe_allow_html=True)

with top_h2:
    if is_dark:
        sun_svg = """
        <div style="display:inline-block; vertical-align:middle; margin-right:6px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffe066" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5"></circle>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
                <path d="M10.5 9.5a2.2 2.2 0 0 1 2.2-0.5" stroke="#ffffff" stroke-width="1.3"></path>
            </svg>
        </div>
        """
        btn_text = "Light Mode"
    else:
        sun_svg = """
        <div style="display:inline-block; vertical-align:middle; margin-right:6px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5" fill="#ea580c"></circle>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
            </svg>
        </div>
        """
        btn_text = "Dark Mode"

    col_icon, col_btn = st.columns([0.3, 1])
    with col_icon:
        st.markdown(sun_svg, unsafe_allow_html=True)
    with col_btn:
        if st.button(btn_text, key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme_mode = "Light" if is_dark else "Dark"
            st.rerun()

# ==============================================================================
# 5. TRIGGER WORD HIGHLIGHTER & SUGGESTION GENERATOR
# ==============================================================================
def highlight_trigger_words(sentence, trigger_words):
    highlighted = sentence
    for tw in sorted(trigger_words, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(tw)})\b", re.IGNORECASE)
        highlighted = pattern.sub(
            r"<mark style='background:rgba(255,119,0,0.38); color:#ffe066; font-weight:800; padding:1px 6px; border-radius:4px; border:1px solid #ffaa00;'>\1</mark>",
            highlighted
        )
    return highlighted

def generate_actionable_suggestion(sentence, paper_name):
    s_low = sentence.lower()
    if any(k in s_low for k in ["complexity", "quadratic", "computation", "long document", "memory footprint", "tpu", "gpu", "resource"]):
        return {
            "type": "⚡ Computational & Scalability Optimization",
            "suggestion": "Develop linear-time kernel approximations (FlashAttention/Performer) or implement sparse attention routing."
        }
    elif any(k in s_low for k in ["imbalance", "annotated", "sample", "dataset", "few", "micro-structure", "lack of data"]):
        return {
            "type": "📊 Data Scarcity & Class Imbalance Remediation",
            "suggestion": "Integrate class-rebalancing loss formulations (Focal Loss) and evaluate diffusion/GAN synthetic oversampling."
        }
    elif any(k in s_low for k in ["hallucination", "factual", "bias", "reasoning", "verification", "external"]):
        return {
            "type": "🛡 Factuality & Grounded Verification",
            "suggestion": "Ground model outputs using Retrieval-Augmented Generation (RAG) and automated self-consistency verification passes."
        }
    elif any(k in s_low for k in ["theoretical", "understanding", "degradation", "skip", "layer", "depth"]):
        return {
            "type": "📐 Architectural & Theoretical Investigation",
            "suggestion": "Formulate dynamical systems bounds for gradient flow or investigate continuous state-space models (Mamba/S4)."
        }
    else:
        return {
            "type": "🔬 Empirical Validation & Benchmark Generalization",
            "suggestion": "Execute cross-domain ablation benchmarks, test out-of-distribution scenarios, and establish standardized evaluation protocols."
        }

# ==============================================================================
# 6. BENCHMARK CORPUS GENERATOR (CROSS-DOMAIN: AI, BIO, MEDICINE, PHYSICS)
# ==============================================================================
def load_benchmark_corpus():
    benchmark_docs = {
        "Vaswani_Transformer_AI_2017.pdf": (
            "We propose the Transformer, a model architecture relying entirely on self-attention for sequence transduction. "
            "However, the computational complexity of self-attention scales quadratically with sequence length, which poses a severe limitation for ultra-long documents. "
            "Future work will investigate sparse attention matrices to extend this to multi-gigabyte document analysis. "
            "A notable drawback remains the lack of recurrence, making positional encoding heuristics necessary."
        ),
        "AlphaFold_Protein_Structure_BioAI_2021.docx": (
            "Accurate computational prediction of 3D protein structures from genetic amino acid sequences has been an elusive challenge in biology. "
            "We introduce AlphaFold, a neural system achieving atomic-scale accuracy across complex macromolecular polymers and cellular assemblies. "
            "Nevertheless, a fundamental limitation is the inability to predict disordered protein dynamics and ligand binding conformational changes. "
            "Furthermore, high GPU memory requirements create a prohibitive bottleneck for multi-chain megadalton complexes. "
            "Future research should explore co-folding with small-molecule therapeutics."
        ),
        "He_Deep_Residual_Learning_ResNet_2016.txt": (
            "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks. "
            "Comprehensive empirical evidence shows that these residual networks are easier to optimize and gain accuracy from greatly increased depth. "
            "However, the memory footprint increases linearly with layer count, which limits deployment on real-time embedded systems. "
            "There remains a lack of theoretical understanding regarding why degradation occurs beyond two hundred layers. "
            "Future investigations must address optimal skip-connection topologies."
        ),
        "CRISPR_Cas9_Genome_Editing_Biology_2020.html": (
            "Targeted genome engineering utilizing RNA-guided Cas9 endonucleases has revolutionized biotechnology, molecular medicine, and cell therapy. "
            "By hybridizing custom single-guide RNA sequences, double-strand breaks can be introduced into cellular DNA with high site-specificity. "
            "However, our method is constrained by off-target cleavage mutations and cytotoxicity in primary human hematopoietic stem cells. "
            "A persistent challenge remains the in vivo delivery across the blood-brain barrier for central nervous system therapies. "
            "Future work will evaluate engineered base editors to minimize double-strand chromosomal breaks."
        ),
        "Quantum_Computing_Qubit_Decoherence_Physics_2022.json": (
            "Superconducting qubits offer a promising path toward fault-tolerant quantum supremacy and polynomial speedups in molecular simulation. "
            "We demonstrate a programmable quantum processor containing 53 superconducting qubits operating at cryogenic millikelvin temperatures. "
            "A severe limitation remains thermal decoherence and high crosstalk noise during simultaneous two-qubit gate operations. "
            "Furthermore, error mitigation requires thousands of physical qubits per logical qubit, which remains incomputable with current cryostats. "
            "Future investigations must address 3D topological surface codes to sustain coherence times."
        )
    }

    history_entries = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_sents = []

    for fname, text in benchmark_docs.items():
        doc_info = extract_universal_document(text.encode("utf-8"), fname)
        st.session_state.papers[fname] = doc_info
        all_sents.extend(doc_info["sentences"])
        history_entries.append({
            "filename": fname,
            "uploaded_at": now_str,
            "format": doc_info["extension"],
            "pages": doc_info["page_count"],
            "words": doc_info["word_count"],
            "sentences": len(doc_info["sentences"])
        })

    # Trigger Self-Learning Vocabulary Induction
    discovered_words = induce_gap_vocabulary_from_corpus(all_sents)
    st.session_state.newly_learned_words = discovered_words

    save_to_history(history_entries)
    st.toast("⚡ Multi-Domain Corpus (AI, Bio, Medicine, Physics) Loaded & Gaps Learned!", icon="✨")

# ==============================================================================
# 7. GLOWING PIPELINE & INSTRUCTIONS COMPONENT
# ==============================================================================
def render_glowing_pipeline_component():
    st.markdown("### Interactive Research Pipeline")
    st.markdown("""
    <div class="pipeline-screenshot-container">
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 1</div>
            <div class="stage-title">Multi-Format Ingestion</div>
            <div class="stage-desc">PDF, DOCX, TXT, JSON, HTML</div>
        </div>
        <div class="pipeline-arrow-glow">➜</div>
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 2</div>
            <div class="stage-title">NLP Cleaning</div>
            <div class="stage-desc">Academic noise removal</div>
        </div>
        <div class="pipeline-arrow-glow">➜</div>
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 3</div>
            <div class="stage-title">Domain Discovery</div>
            <div class="stage-desc">Bio, AI, Med, Physics</div>
        </div>
        <div class="pipeline-arrow-glow">➜</div>
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 4</div>
            <div class="stage-title">Cross-Clustering</div>
            <div class="stage-desc">PCA & Topic Overlaps</div>
        </div>
        <div class="pipeline-arrow-glow">➜</div>
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 5</div>
            <div class="stage-title">Adaptive Gap Mining</div>
            <div class="stage-desc">Self-Learning + 1D-CNN</div>
        </div>
        <div class="pipeline-arrow-glow">➜</div>
        <div class="pipeline-stage-box">
            <div class="stage-num">Stage 6</div>
            <div class="stage-title">Executive Dossier</div>
            <div class="stage-desc">ReportLab A4 PDF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 8. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown('<p class="brand-sub">Research Intelligence Platform</p>', unsafe_allow_html=True)

    page = st.radio(
        "Workspace",
        [
            "🏠 Home",
            "🚀 Upload & Analyze Hub",
            "📄 Multi-Format Papers Library",
            "📈 Text Analysis & EDA",
            "🔗 Cosine Similarity",
            "🧩 Interdisciplinary Clustering",
            "🔍 Adaptive Gap Mining & Suggestions",
            "🧠 Learned Vocabulary & AI Rules",
            "📑 Dossier Reports",
            "🔬 Pipeline Architecture",
            "❓ FAQ & Research Q&A",
            "⚙ Settings & Platform"
        ],
        key="nav_selection",
        label_visibility="collapsed"
    )

    st.markdown("---")
    active_papers_count = len(st.session_state.papers)
    st.caption("ACTIVE TELEMETRY")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Documents", f"{active_papers_count}")
    with col_sb2:
        tot_sents = sum(len(p.get("sentences", [])) for p in st.session_state.papers.values())
        st.metric("Sentences", f"{tot_sents}")

    st.markdown("---")
    if st.button("⚡ Load Multi-Domain Corpus", use_container_width=True):
        load_benchmark_corpus()
        st.rerun()

    if active_papers_count > 0:
        if st.button("🗑 Reset Current Session", use_container_width=True, type="secondary"):
            st.session_state.papers = {}
            st.session_state.validated_gaps = {}
            st.session_state.rejected_gaps = set()
            st.session_state.newly_learned_words = []
            st.rerun()

    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; margin-top:22px; text-align:center; font-weight:600; letter-spacing:0.04em;">
        ResearchLens AI @ 2026
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 9. ROUTING: 🏠 HOME PAGE
# ==============================================================================
if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">Welcome to ResearchLens AI</div>
        <div class="hero-subtitle">
            An Autonomous Cross-Domain Research Intelligence Platform engineered to ingest scientific literature 
            across all formats (PDF, Word DOCX, TXT, MD, CSV, JSON, HTML), model interdisciplinary clusters 
            (Biology, AI, Clinical Medicine, Physics, Chemistry), and mine research gaps using an adaptive self-learning 
            engine that continuously discovers new gap indicators over time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    hcol1, hcol2, hcol3 = st.columns([1.3, 1.3, 2])
    with hcol1:
        if st.button("🚀 Load Multi-Domain Corpus", use_container_width=True):
            load_benchmark_corpus()
            st.rerun()
    with hcol2:
        if st.button("📂 Upload Files (All Formats)", use_container_width=True, type="secondary"):
            st.session_state.nav_selection = "🚀 Upload & Analyze Hub"
            st.rerun()

    st.markdown("---")
    render_glowing_pipeline_component()

# ==============================================================================
# 10. ROUTING: 🚀 UPLOAD & ANALYZE HUB (ALL FORMATS + CROSS-DOMAIN SCAN)
# ==============================================================================
elif page == "🚀 Upload & Analyze Hub":
    st.title("One-Click Literature Analysis Hub")
    st.caption("Upload papers in any format (PDF, DOCX, TXT, MD, CSV, JSON, HTML) for automated cross-domain analysis.")

    with st.container(border=True):
        st.subheader("1. Ingest Research Papers & Documents")
        uploaded_files = st.file_uploader(
            "Upload any files (PDF, Word DOCX, Text, Markdown, CSV, JSON, HTML)",
            type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm"],
            accept_multiple_files=True,
            help="Supports all document types simultaneously. Automatically strips bibliographies and noise."
        )

        if uploaded_files:
            progress_bar = st.progress(0, text="Extracting and indexing multi-format literature...")
            new_history_entries = []
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            all_new_sents = []

            for idx, uf in enumerate(uploaded_files):
                if uf.name not in st.session_state.papers:
                    try:
                        doc_info = extract_universal_document(uf.read(), uf.name)
                        st.session_state.papers[uf.name] = doc_info
                        all_new_sents.extend(doc_info["sentences"])

                        new_history_entries.append({
                            "filename": uf.name,
                            "uploaded_at": now_str,
                            "format": doc_info["extension"],
                            "pages": doc_info["page_count"],
                            "words": doc_info["word_count"],
                            "sentences": len(doc_info["sentences"])
                        })
                    except Exception as err:
                        st.error(f"Error parsing {uf.name}: {err}")

                progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processed: {uf.name}")

            # Run Continuous Vocabulary Induction
            learned_words = induce_gap_vocabulary_from_corpus(all_new_sents)
            if learned_words:
                st.session_state.newly_learned_words = learned_words
                st.toast(f"🧠 Model Learned {len(learned_words)} New Domain Gap Words!", icon="✨")

            if new_history_entries:
                save_to_history(new_history_entries)
                st.success(f"✓ {len(new_history_entries)} documents ingested across multiple formats!")

    if not st.session_state.papers:
        st.info("No documents currently loaded. Upload files above or click 'Load Multi-Domain Corpus' in the sidebar.")
    else:
        st.markdown("---")
        st.subheader("2. Executive Cross-Domain Telemetry")
        
        num_p = len(st.session_state.papers)
        tot_w = sum(p.get("word_count", 0) for p in st.session_state.papers.values())
        tot_s = sum(len(p.get("sentences", [])) for p in st.session_state.papers.values())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Ingested", f"{num_p}")
        with col2:
            st.metric("Total Words Extracted", f"{tot_w:,}")
        with col3:
            st.metric("Syntactic Claims", f"{tot_s:,}")
        with col4:
            st.metric("Learned Gap Words", f"{len(st.session_state.newly_learned_words)}", delta="Adaptive")

        st.markdown("---")
        st.subheader("3. Automated Cross-Domain Synthesis")
        
        hub_tab1, hub_tab2, hub_tab3 = st.tabs(["Discovered Domain Badges", "Similarity Heatmap", "Mined Gaps & Discovered Words"])

        with hub_tab1:
            st.markdown("#### Document Disciplinary Classification")
            st.caption("ℹ️ *'Total Word Count' reflects the extracted paper length (excluding bibliography/references). Primary discipline is inferred using cross-domain ontological matching.*")
            dom_rows = []
            for name, meta in st.session_state.papers.items():
                primary, overlaps = detect_document_domains(meta["text"])
                dom_rows.append({
                    "Document Name": name,
                    "Format": meta.get("extension", "PDF"),
                    "Primary Discipline": primary,
                    "Cross-Disciplinary Overlaps": ", ".join(overlaps) if overlaps else "Core Domain",
                    "Total Word Count (Paper Length)": f"{meta['word_count']:,}"
                })
            st.dataframe(pd.DataFrame(dom_rows), use_container_width=True)

        with hub_tab2:
            if st.session_state.papers:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                paper_names = list(st.session_state.papers.keys())
                corpus = [st.session_state.papers[p]["text"] for p in paper_names]

                has_words = any(len(re.findall(r'\b[a-zA-Z]{2,}\b', text)) > 0 for text in corpus)
                if not has_words:
                    st.warning("⚠️ The uploaded document(s) contain insufficient text vocabulary for TF-IDF calculations. Please ensure documents contain extractable English text.")
                else:
                    try:
                        vec = TfidfVectorizer(stop_words='english', max_features=300, token_pattern=r'(?u)\b\w+\b')
                        tfidf_matrix = vec.fit_transform(corpus)
                        sim_matrix = cosine_similarity(tfidf_matrix)

                        df_sim = pd.DataFrame(sim_matrix, index=[p[:18] for p in paper_names], columns=[p[:18] for p in paper_names])

                        plot_w = max(5.0, min(10.0, len(paper_names) * 1.2))
                        plot_h = max(3.5, min(8.0, len(paper_names) * 1.0))
                        fig_sim, ax_sim = plt.subplots(figsize=(plot_w, plot_h), facecolor=bg_color)
                        ax_sim.set_facecolor(bg_color)
                        sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax_sim, cbar=True,
                                    annot_kws={"color": "#080c14", "fontweight": "bold"})
                        ax_sim.tick_params(colors=text_color, labelsize=8)
                        ax_sim.set_title("Cross-Domain Semantic Similarity Matrix", color=text_color, fontsize=11, fontweight="bold")
                        st.pyplot(fig_sim)

                        if len(paper_names) == 1:
                            st.info("💡 1 document loaded (Self-similarity = 1.00). Upload additional papers to view cross-document similarity heatmaps.")
                    except ValueError as ve:
                        st.warning(f"⚠️ Could not compute TF-IDF similarity: {ve}. The documents may have insufficient vocabulary.")
            else:
                st.info("Upload at least 1 document to view similarity heatmaps.")

        with hub_tab3:
            active_triggers = load_learned_triggers()

            # 1. Discovered Vocabulary & Learned Gap Indicators Section
            st.markdown("#### 🧬 Discovered Vocabulary & Adaptive Gap Triggers")
            if st.session_state.newly_learned_words:
                st.success(f"✨ **{len(st.session_state.newly_learned_words)} New Gap Triggers** automatically learned & inducted from your uploaded papers:")
                st.write(" ".join([f"<span class='gap-score-badge' style='margin-right:6px; margin-bottom:6px; display:inline-block;'>✨ {w}</span>" for w in st.session_state.newly_learned_words]), unsafe_allow_html=True)
            else:
                st.info(f"💡 The dynamic learner continuously scans papers for limitation patterns across AI, Bio, Medicine, Physics, and Chem. Currently active triggers in memory: **{len(active_triggers)} words**.")
                with st.expander("🔍 View Active Gap Trigger Lexicon"):
                    st.write(", ".join([f"`{w}`" for w in sorted(list(active_triggers))[:40]]) + ("..." if len(active_triggers) > 40 else ""))

            st.markdown("---")

            # 2. Vectorized Batch Mining of Gaps
            st.markdown("#### 🎯 Mined Research Gaps & Actionable Recommendations")
            
            # Interactive Confidence Filter
            f_col1, f_col2 = st.columns([2, 3])
            with f_col1:
                conf_threshold = st.slider("Min Gap Confidence Filter:", min_value=0.20, max_value=0.85, value=0.38, step=0.02, help="Lower to see more candidate gaps, raise for only highest-confidence limitations.")

            # Prepare sentences across all loaded papers
            candidate_items = []
            for doc_name, meta in st.session_state.papers.items():
                for sent in meta.get("sentences", []):
                    candidate_items.append((doc_name, sent))

            gaps_found = []
            if candidate_items:
                all_sents = [item[1] for item in candidate_items]
                # Fast vectorized batch evaluation
                cnn_probs = batch_predict_gap_probabilities(all_sents)

                for (doc_name, sent), cnn_prob in zip(candidate_items, cnn_probs):
                    s_low = sent.lower()
                    matched_rules = [kw for kw in active_triggers if kw in s_low]
                    rule_score = min(1.0, len(matched_rules) * 0.40) if matched_rules else 0.0

                    # Balanced hybrid confidence score
                    if matched_rules and cnn_prob >= 0.30:
                        hybrid_score = round(0.5 * rule_score + 0.5 * cnn_prob, 3)
                    elif matched_rules:
                        hybrid_score = round(0.6 * rule_score + 0.4 * cnn_prob, 3)
                    else:
                        hybrid_score = round(cnn_prob, 3)

                    if hybrid_score >= conf_threshold:
                        suggestion_info = generate_actionable_suggestion(sent, doc_name)
                        gap_id = f"{doc_name}_{hash(sent)}"
                        gaps_found.append({
                            "id": gap_id,
                            "paper": doc_name,
                            "sentence": sent,
                            "rules": matched_rules,
                            "cnn_prob": cnn_prob,
                            "hybrid_score": hybrid_score,
                            "suggestion_type": suggestion_info["type"],
                            "suggestion_text": suggestion_info["suggestion"]
                        })

            gaps_found.sort(key=lambda x: x["hybrid_score"], reverse=True)

            if gaps_found:
                st.write(f"Identified **{len(gaps_found)} research gap claims** passing confidence threshold ≥ `{conf_threshold:.2f}`:")
                for g in gaps_found:
                    gid = g["id"]
                    if gid in st.session_state.rejected_gaps:
                        continue

                    is_validated = gid in st.session_state.validated_gaps
                    highlighted_html = highlight_trigger_words(g['sentence'], g['rules'])

                    trigger_desc = f"Indicators: {', '.join(g['rules'])}" if g['rules'] else "CNN Neural Sequence Bottleneck"

                    st.markdown(f"""
                    <div class="gap-card">
                        <div class="gap-paper">📄 Source: {g['paper']}</div>
                        <div class="gap-text">"{highlighted_html}"</div>
                        <div>
                            <span class="gap-score-badge">Hybrid Score: {g['hybrid_score']}</span>
                            <span class="gap-score-badge" style="border-color:#38bdf8; color:#38bdf8;">CNN Structural Score: {g['cnn_prob']*100:.1f}%</span>
                            <span class="gap-score-badge" style="border-color:#a855f7; color:#c084fc;">{trigger_desc}</span>
                        </div>
                        <div class="gap-suggestion-box">
                            <div class="gap-suggestion-title">💡 Actionable Research Recommendation:</div>
                            <div style="font-weight:700; font-size:0.8rem; color:#10b981; margin-bottom:2px;">{g['suggestion_type']}</div>
                            <div class="gap-suggestion-desc">{g['suggestion_text']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    bcol1, bcol2, bcol3 = st.columns([1.2, 1, 4])
                    with bcol1:
                        if not is_validated:
                            if st.button("✓ Validate Gap", key=f"hub_val_{gid}"):
                                st.session_state.validated_gaps[gid] = g
                                st.rerun()
                        else:
                            st.success("✓ Validated by Researcher")
                    with bcol2:
                        if st.button("✕ Reject", key=f"hub_rej_{gid}"):
                            st.session_state.rejected_gaps.add(gid)
                            if gid in st.session_state.validated_gaps:
                                del st.session_state.validated_gaps[gid]
                            st.rerun()

            # Instant Download Card when gaps are validated
            if st.session_state.validated_gaps:
                st.markdown("---")
                with st.container(border=True):
                    st.markdown("### 📥 Download Modified Document (Addressed Gaps Applied)")
                    st.caption("Download the updated paper enriched with the formal 'Validated Research Gaps & Strategic Amendments' section.")

                    doc_groups = {}
                    for g in st.session_state.validated_gaps.values():
                        doc_groups.setdefault(g.get("paper", ""), []).append(g)

                    c_doc, c_fmt, c_btn = st.columns([2.5, 2, 2])
                    with c_doc:
                        target_doc = st.selectbox("Document to Download:", options=list(doc_groups.keys()), key="hub_exp_doc")
                    with c_fmt:
                        target_fmt = st.selectbox("Format:", options=["PDF (.pdf)", "Word Document (.docx)", "Markdown (.md)", "Plain Text (.txt)", "JSON (.json)", "HTML (.html)"], key="hub_exp_fmt")
                    with c_btn:
                        st.write("")
                        st.write("")
                        orig_text = st.session_state.papers.get(target_doc, {}).get("text", "")
                        doc_gaps = doc_groups[target_doc]
                        buf, out_name, mime_type = export_modified_file(target_doc, orig_text, doc_gaps, target_fmt)
                        st.download_button(
                            label=f"💾 Download Modified {target_fmt.split()[0]}",
                            data=buf,
                            file_name=out_name,
                            mime=mime_type,
                            type="primary",
                            use_container_width=True
                        )
            else:
                st.warning(f"No research gaps scored above the threshold of {conf_threshold:.2f}. Try lowering the confidence slider above.")

# ==============================================================================
# 11. ROUTING: 📄 MULTI-FORMAT PAPERS LIBRARY
# ==============================================================================
elif page == "📄 Multi-Format Papers Library":
    st.title("Multi-Format Research Library & Audit Log")
    st.caption("Upload and manage documents in PDF, DOCX, TXT, MD, CSV, JSON, and HTML formats.")

    with st.container(border=True):
        st.subheader("Upload Research Documents (Universal Ingestion)")
        uploaded_files = st.file_uploader(
            "Select or drop files of any type",
            type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm"],
            accept_multiple_files=True
        )

        if uploaded_files:
            progress_bar = st.progress(0, text="Extracting documents...")
            new_history_entries = []
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            for idx, uf in enumerate(uploaded_files):
                if uf.name not in st.session_state.papers:
                    try:
                        doc_info = extract_universal_document(uf.read(), uf.name)
                        st.session_state.papers[uf.name] = doc_info

                        new_history_entries.append({
                            "filename": uf.name,
                            "uploaded_at": now_str,
                            "format": doc_info["extension"],
                            "pages": doc_info["page_count"],
                            "words": doc_info["word_count"],
                            "sentences": len(doc_info["sentences"])
                        })
                    except Exception as err:
                        st.error(f"Error extracting {uf.name}: {err}")

                progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processed ({idx+1}/{len(uploaded_files)}): {uf.name}")

            if new_history_entries:
                save_to_history(new_history_entries)
                st.success(f"Successfully processed and archived {len(new_history_entries)} new documents!")

    if st.session_state.papers:
        st.subheader("Documents Currently in Memory")
        st.caption("Click ❌ next to any paper to immediately unload it from active session memory.")
        
        m1, m2, m3, m4, m5 = st.columns([3.5, 2.5, 1.2, 1.2, 0.8])
        m1.markdown("**Document Name**")
        m2.markdown("**Discipline**")
        m3.markdown("**Words**")
        m4.markdown("**Sentences**")
        m5.markdown("**Action**")
        st.markdown("<hr style='margin:4px 0 10px 0; border-color:rgba(255,119,0,0.3);'>", unsafe_allow_html=True)

        for p_idx, (name, meta) in enumerate(list(st.session_state.papers.items())):
            primary, _ = detect_document_domains(meta["text"])
            c1, c2, c3, c4, c5 = st.columns([3.5, 2.5, 1.2, 1.2, 0.8])
            c1.markdown(f"📄 **{name}** `({meta.get('extension', 'PDF')})`")
            c2.caption(primary)
            c3.write(f"{meta['word_count']:,}")
            c4.write(str(len(meta["sentences"])))
            if c5.button("❌", key=f"unload_mem_{p_idx}", help=f"Unload '{name}' from current memory"):
                st.session_state.papers.pop(name, None)
                st.rerun()
            st.markdown("<hr style='margin:2px 0 6px 0; border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📜 Historical Uploads Archive (All Formats)")
    history_records = load_history()
    if history_records:
        st.caption("Click the 🗑️ icon beside any document to permanently remove it from history.")
        
        h1, h2, h3, h4, h5, h6 = st.columns([3.5, 1.8, 1, 1.2, 1.2, 0.8])
        h1.markdown("**Filename**")
        h2.markdown("**Uploaded At**")
        h3.markdown("**Pages**")
        h4.markdown("**Words**")
        h5.markdown("**Sentences**")
        h6.markdown("**Action**")
        st.markdown("<hr style='margin:4px 0 10px 0; border-color:rgba(255,119,0,0.3);'>", unsafe_allow_html=True)

        for idx, r in enumerate(history_records):
            c1, c2, c3, c4, c5, c6 = st.columns([3.5, 1.8, 1, 1.2, 1.2, 0.8])
            fname = r.get("filename", "Unknown")
            c1.markdown(f"📄 `{fname}`")
            c2.caption(r.get("uploaded_at", "-"))
            c3.write(str(r.get("pages", 1)))
            c4.write(f"{r.get('words', 0):,}")
            c5.write(str(r.get("sentences", 0)))
            if c6.button("🗑️", key=f"del_hist_row_{idx}", help=f"Remove '{fname}' from history"):
                delete_from_history([fname])
                st.rerun()
            st.markdown("<hr style='margin:2px 0 6px 0; border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    else:
        st.info("No past upload history found yet.")

# ==============================================================================
# 12. ROUTING: 📈 TEXT ANALYSIS & EDA
# ==============================================================================
elif page == "📈 Text Analysis & EDA":
    st.title("Exploratory Data Analysis & Text Metrics")
    st.caption("Distribution of vocabulary, token frequencies, and linguistic features across disciplines.")

    if not st.session_state.papers:
        st.info("Please load papers or the multi-domain corpus first to unlock EDA.")
    else:
        all_text = " ".join([p["text"] for p in st.session_state.papers.values()])
        stopwords = set(["the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "on", "as", "are", "by", "an", "be", "we", "from", "which", "our"])
        raw_tokens = [w.lower().strip(".,;:()[]\"'") for w in all_text.split() if len(w) > 3]
        clean_tokens_list = [w for w in raw_tokens if w not in stopwords and w.isalpha()]

        from collections import Counter
        counts = Counter(clean_tokens_list).most_common(12)
        df_counts = pd.DataFrame(counts, columns=["Term", "Frequency"])

        col_chart, col_data = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor=bg_color)
            ax.set_facecolor(bg_color)
            bars = ax.barh(df_counts["Term"][::-1], df_counts["Frequency"][::-1], color="#ff7700", edgecolor="#ffe066", linewidth=1.5, alpha=0.9)
            ax.tick_params(colors=text_color, labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#ff7700")
            ax.spines['bottom'].set_color("#ff7700")
            ax.set_title("Most Frequent Research Terms Across Corpus", color=text_color, fontsize=11, fontweight="bold", pad=12)
            st.pyplot(fig)
        with col_data:
            st.dataframe(df_counts, use_container_width=True)

# ==============================================================================
# 13. ROUTING: 🔗 COSINE SIMILARITY
# ==============================================================================
elif page == "🔗 Cosine Similarity":
    st.title("Cross-Domain Semantic Similarity")
    st.caption("Pairwise Cosine Similarity computed over TF-IDF vector representations.")

    if not st.session_state.papers:
        st.warning("Cosine similarity requires at least 1 paper in the library. Please load benchmark corpus or upload papers.")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        paper_names = list(st.session_state.papers.keys())
        corpus = [st.session_state.papers[p]["text"] for p in paper_names]

        has_words = any(len(re.findall(r'\b[a-zA-Z]{2,}\b', text)) > 0 for text in corpus)
        if not has_words:
            st.warning("⚠️ The uploaded document(s) contain insufficient text vocabulary for Cosine Similarity. Please ensure documents contain extractable English text.")
        else:
            try:
                vec = TfidfVectorizer(stop_words='english', max_features=500, token_pattern=r'(?u)\b\w+\b')
                tfidf_matrix = vec.fit_transform(corpus)
                sim_matrix = cosine_similarity(tfidf_matrix)

                df_sim = pd.DataFrame(sim_matrix, index=[p[:20] for p in paper_names], columns=[p[:20] for p in paper_names])

                plot_w = max(5.5, min(10.0, len(paper_names) * 1.2))
                plot_h = max(4.0, min(8.0, len(paper_names) * 1.0))

                col_heat, col_pairs = st.columns([3, 2])
                with col_heat:
                    fig, ax = plt.subplots(figsize=(plot_w, plot_h), facecolor=bg_color)
                    ax.set_facecolor(bg_color)
                    sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax, cbar=True,
                                annot_kws={"color": "#080c14", "fontweight": "bold"})
                    ax.tick_params(colors=text_color, labelsize=8)
                    ax.set_title("Cosine Similarity Matrix", color=text_color, fontsize=11, fontweight="bold", pad=12)
                    st.pyplot(fig)

                with col_pairs:
                    st.markdown("#### Top Similar Paper Pairs")
                    if len(paper_names) >= 2:
                        pairs = []
                        for i in range(len(paper_names)):
                            for j in range(i + 1, len(paper_names)):
                                pairs.append((paper_names[i], paper_names[j], sim_matrix[i, j]))
                        pairs.sort(key=lambda x: x[2], reverse=True)

                        for p1, p2, score in pairs[:6]:
                            st.markdown(f"""
                            <div class="kpi-card" style="margin-bottom:12px; border-width: 2px;">
                                <div style="font-size:0.75rem; color:#ffaa00; font-weight:800;">SIMILARITY: {score:.2f}</div>
                                <div style="font-size:0.84rem; font-weight:700; color:{text_color}; margin-top:2px;">{p1}</div>
                                <div style="font-size:0.75rem; color:{sub_text};">↔ {p2}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("💡 1 document loaded (Self-similarity = 1.00). Cross-paper comparison pairs will automatically populate when you upload more papers.")
            except ValueError as ve:
                st.warning(f"⚠️ Could not compute TF-IDF Cosine Similarity: {ve}")

# ==============================================================================
# 14. ROUTING: 🧩 INTERDISCIPLINARY CLUSTERING (ALL DOMAINS: BIO, AI, MED, PHYS)
# ==============================================================================
elif page == "🧩 Interdisciplinary Clustering":
    st.title("Interdisciplinary Multi-Domain Clustering")
    st.caption("Partition complex cross-domain literature (Biology, AI, Clinical Medicine, Physics, Chemistry) into clusters with 2D PCA spatial maps.")

    if not st.session_state.papers:
        st.warning("Clustering requires at least 1 document. Please upload or load literature.")
    else:
        num_docs = len(st.session_state.papers)
        if num_docs == 1:
            k_val = 1
            st.info("ℹ️ 1 document loaded. Displaying single-document topological representation (Cluster 1). Upload more papers to partition into multiple clusters.")
        elif num_docs == 2:
            k_val = 2
            st.info("ℹ️ 2 documents loaded. Partitioned into k = 2 clusters.")
        else:
            max_k = min(15, num_docs)
            k_val = st.slider("Select Cluster Count (k)", min_value=2, max_value=max_k, value=min(3, max_k), help="Dynamically scales for any number of uploaded documents.")
        clust_res = perform_universal_clustering(st.session_state.papers, k=k_val)

        if "error" in clust_res:
            st.error(clust_res["error"])
        else:
            df_pca = clust_res["df_clusters"]
            topics = clust_res["cluster_topics"]

            col_plot, col_clusters = st.columns([3, 2])
            with col_plot:
                fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=bg_color)
                ax.set_facecolor(bg_color)
                colors = ["#ff7700", "#38bdf8", "#ec4899", "#10b981", "#a855f7"]

                for idx, c_label in enumerate(sorted(df_pca["Cluster"].unique())):
                    sub = df_pca[df_pca["Cluster"] == c_label]
                    ax.scatter(sub["x"], sub["y"], label=c_label, s=220, color=colors[idx % len(colors)],
                               edgecolors="#ffffff", linewidths=1.8, alpha=0.95)
                    for _, row in sub.iterrows():
                        ax.annotate(row["Paper"][:14], (row["x"] + 0.02, row["y"] + 0.02), color=text_color, fontsize=8, weight="bold")

                ax.tick_params(colors=sub_text)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color("#ff7700")
                ax.spines['bottom'].set_color("#ff7700")
                ax.set_title("2D PCA Cross-Disciplinary Document Map", color=text_color, fontsize=11, fontweight="bold")
                ax.legend(facecolor=bg_color, edgecolor="#ff7700", labelcolor=text_color)
                st.pyplot(fig)

            with col_clusters:
                st.markdown("#### Discovered Thematic Topics & Domain Badges")
                for c_id, kws in topics.items():
                    st.markdown(f"**{c_id}**")
                    st.caption(f"Key Terms: {', '.join(kws)}")
                    sub_papers = df_pca[df_pca["Cluster"] == c_id]
                    for _, row in sub_papers.iterrows():
                        st.markdown(f"• `{row['Paper'][:22]}` ➔ *{row['Domain']}*")

# ==============================================================================
# 15. ROUTING: 🔍 ADAPTIVE GAP MINING & SUGGESTIONS
# ==============================================================================
elif page == "🔍 Adaptive Gap Mining & Suggestions":
    st.title("Adaptive Gap Mining & Discovered Indicator Attribution")
    st.caption("Hybrid Gap Detection using continuous vocabulary learning and sequence-level 1D-CNN inference.")

    if not st.session_state.papers:
        st.info("No documents found. Load the multi-domain corpus to mine potential research gaps.")
    else:
        active_triggers = load_learned_triggers()

        # Interactive Confidence Filter
        f_col1, f_col2 = st.columns([2, 3])
        with f_col1:
            conf_threshold = st.slider("Confidence Sensitivity Filter:", min_value=0.20, max_value=0.85, value=0.38, step=0.02, key="adaptive_page_slider")

        candidate_items = []
        for doc_name, meta in st.session_state.papers.items():
            for sent in meta.get("sentences", []):
                candidate_items.append((doc_name, sent))

        gaps_found = []
        if candidate_items:
            all_sents = [item[1] for item in candidate_items]
            cnn_probs = batch_predict_gap_probabilities(all_sents)

            for (doc_name, sent), cnn_prob in zip(candidate_items, cnn_probs):
                s_low = sent.lower()
                matched_rules = [kw for kw in active_triggers if kw in s_low]
                rule_score = min(1.0, len(matched_rules) * 0.40) if matched_rules else 0.0

                if matched_rules and cnn_prob >= 0.30:
                    hybrid_score = round(0.5 * rule_score + 0.5 * cnn_prob, 3)
                elif matched_rules:
                    hybrid_score = round(0.6 * rule_score + 0.4 * cnn_prob, 3)
                else:
                    hybrid_score = round(cnn_prob, 3)

                if hybrid_score >= conf_threshold:
                    suggestion_info = generate_actionable_suggestion(sent, doc_name)
                    gap_id = f"{doc_name}_{hash(sent)}"
                    gaps_found.append({
                        "id": gap_id,
                        "paper": doc_name,
                        "sentence": sent,
                        "rules": matched_rules,
                        "cnn_prob": cnn_prob,
                        "hybrid_score": hybrid_score,
                        "suggestion_type": suggestion_info["type"],
                        "suggestion_text": suggestion_info["suggestion"]
                    })

        gaps_found.sort(key=lambda x: x["hybrid_score"], reverse=True)

        if gaps_found:
            st.write(f"Identified **{len(gaps_found)} high-probability limitation claims** across your literature (Threshold ≥ `{conf_threshold:.2f}`):")
        else:
            st.warning(f"No research gaps scored above {conf_threshold:.2f}. Try lowering the confidence slider above.")

        for idx, g in enumerate(gaps_found):
            gid = g["id"]
            if gid in st.session_state.rejected_gaps:
                continue

            is_validated = gid in st.session_state.validated_gaps
            highlighted_html = highlight_trigger_words(g['sentence'], g['rules'])

            st.markdown(f"""
            <div class="gap-card">
                <div class="gap-paper">📄 Source: {g['paper']}</div>
                <div class="gap-text">"{highlighted_html}"</div>
                <div>
                    <span class="gap-score-badge">Hybrid Score: {g['hybrid_score']}</span>
                    <span class="gap-score-badge" style="border-color:#38bdf8; color:#38bdf8;">CNN Structural Probability: {g['cnn_prob']*100:.1f}%</span>
                    <span class="gap-score-badge" style="border-color:#a855f7; color:#c084fc;">Triggers: {', '.join(g['rules']) if g['rules'] else 'CNN Sequence Context'}</span>
                </div>
                <div class="gap-suggestion-box">
                    <div class="gap-suggestion-title">💡 Actionable Research Recommendation:</div>
                    <div style="font-weight:700; font-size:0.8rem; color:#10b981; margin-bottom:2px;">{g['suggestion_type']}</div>
                    <div class="gap-suggestion-desc">{g['suggestion_text']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            bcol1, bcol2, bcol3 = st.columns([1.2, 1, 4])
            with bcol1:
                if not is_validated:
                    if st.button("✓ Validate Gap", key=f"val_{gid}"):
                        st.session_state.validated_gaps[gid] = g
                        st.rerun()
                else:
                    st.success("✓ Validated by Researcher")
            with bcol2:
                if st.button(f"✕ Reject", key=f"rej_{gid}"):
                    st.session_state.rejected_gaps.add(gid)
                    if gid in st.session_state.validated_gaps:
                        del st.session_state.validated_gaps[gid]
                    st.rerun()

        if st.session_state.validated_gaps:
            st.markdown("---")
            with st.container(border=True):
                st.markdown("### 📥 Download Modified Document (Addressed Gaps Applied)")
                st.caption("Download the updated paper enriched with the formal 'Validated Research Gaps & Strategic Amendments' section.")

                doc_groups = {}
                for g in st.session_state.validated_gaps.values():
                    doc_groups.setdefault(g.get("paper", ""), []).append(g)

                c_doc, c_fmt, c_btn = st.columns([2.5, 2, 2])
                with c_doc:
                    target_doc = st.selectbox("Document to Download:", options=list(doc_groups.keys()), key="gap_page_doc")
                with c_fmt:
                    target_fmt = st.selectbox("Choose Format:", options=["PDF (.pdf)", "Word Document (.docx)", "Markdown (.md)", "Plain Text (.txt)", "JSON (.json)", "HTML (.html)"], key="gap_page_fmt")
                with c_btn:
                    st.write("")
                    st.write("")
                    orig_text = st.session_state.papers.get(target_doc, {}).get("text", "")
                    doc_gaps = doc_groups[target_doc]
                    buf, out_name, mime_type = export_modified_file(target_doc, orig_text, doc_gaps, target_fmt)
                    st.download_button(
                        label=f"💾 Download Modified {target_fmt.split()[0]}",
                        data=buf,
                        file_name=out_name,
                        mime=mime_type,
                        type="primary",
                        use_container_width=True
                    )

# ==============================================================================
# 16. ROUTING: 🧠 LEARNED VOCABULARY & AI RULES
# ==============================================================================
elif page == "🧠 Learned Vocabulary & AI Rules":
    st.title("Adaptive Model Learning & Discovered Vocabulary")
    st.caption("How the system learns new gap indicators from text embeddings instead of relying solely on static NLP rules.")

    active_triggers = sorted(list(load_learned_triggers()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Active Learned Gap Trigger Lexicon")
        st.write(f"The model has inducted **{len(active_triggers)} active indicator words** into its memory:")
        st.write(", ".join([f"`{t}`" for t in active_triggers]))

        # Add custom trigger manually
        new_word_input = st.text_input("Teach the Model a New Gap Indicator (Domain Word):", placeholder="e.g. incomputable, unviable, cytotoxic")
        if st.button("➕ Add Word to Model Memory"):
            if new_word_input:
                cur_triggers = load_learned_triggers()
                cur_triggers.add(new_word_input.lower().strip())
                from src.gap_rules import save_learned_triggers
                save_learned_triggers(cur_triggers)
                st.success(f"✓ '{new_word_input}' has been permanently learned by the gap detector!")
                st.rerun()

    with col2:
        st.subheader("🧬 Automated Discovery from Uploaded Papers")
        st.write("When you upload papers, continuous Word2Vec scans your domain text and surfaces new candidate indicator words:")
        if st.session_state.newly_learned_words:
            st.success(f"Discovered {len(st.session_state.newly_learned_words)} new indicators from your papers:")
            st.write(", ".join([f"✨ **{w}**" for w in st.session_state.newly_learned_words]))
        else:
            st.info("Upload papers to trigger automatic vector similarity induction.")

# ==============================================================================
# 17. ROUTING: 📑 DOSSIER REPORTS
# ==============================================================================
elif page == "📑 Dossier Reports":
    st.title("Executive Research Dossier Generator")
    st.caption("Compile your verified cross-domain research gaps, statistics, and literature findings into an A4 PDF document.")

    val_count = len(st.session_state.validated_gaps)
    st.metric("Researcher-Validated Gaps Ready for Export", f"{val_count}")

    if val_count == 0:
        st.warning("You haven't validated any gaps yet. Navigate to '🔍 Adaptive Gap Mining & Suggestions' and click '✓ Validate Gap'.")
    else:
        st.subheader("Summary of Verified Gaps & Recommendations")
        for g in st.session_state.validated_gaps.values():
            st.markdown(f"• **{g['paper']}**: *\"{g['sentence']}\"*")
            st.caption(f"↳ 💡 Recommendation: {g.get('suggestion_text', 'Further empirical benchmarks needed.')}")

        st.markdown("---")
        pdf_buffer = generate_pdf_report(st.session_state.validated_gaps, len(st.session_state.papers))
        
        st.download_button(
            label="📄 Export & Download Comprehensive Research Dossier (PDF)",
            data=pdf_buffer,
            file_name="ResearchLens_CrossDomain_Dossier.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📥 Export Individual Modified Papers (Choose Output Format)")
        st.caption("Select any paper from your library and download its amended version with all validated research gaps incorporated.")

        all_papers_list = list(st.session_state.papers.keys())
        if all_papers_list:
            c_p, c_f = st.columns([2.5, 2])
            with c_p:
                sel_p = st.selectbox("Select Paper to Export:", options=all_papers_list, key="dossier_indiv_doc")
            with c_f:
                sel_f = st.selectbox(
                    "Select Output Format:",
                    options=["PDF (.pdf)", "Word Document (.docx)", "Markdown (.md)", "Plain Text (.txt)", "JSON (.json)", "HTML (.html)"],
                    key="dossier_indiv_fmt"
                )

            p_text = st.session_state.papers[sel_p].get("text", "")
            p_gaps = [g for g in st.session_state.validated_gaps.values() if g.get("paper") == sel_p]

            with st.expander(f"👁 Preview Changes & Amendments for '{sel_p}' ({len(p_gaps)} Validated Gaps)", expanded=False):
                preview_str = build_modified_document_content(p_text, sel_p, p_gaps)
                st.text_area("Enriched Document Preview:", value=preview_str[:3500] + ("\n... [truncated for preview]" if len(preview_str) > 3500 else ""), height=220, disabled=True)

            buf_indiv, out_indiv, mime_indiv = export_modified_file(sel_p, p_text, p_gaps, sel_f)
            st.download_button(
                label=f"💾 Download Modified '{sel_p}' as {sel_f.split()[0]}",
                data=buf_indiv,
                file_name=out_indiv,
                mime=mime_indiv,
                type="primary",
                use_container_width=True
            )

# ==============================================================================
# 18. ROUTING: 🔬 PIPELINE ARCHITECTURE
# ==============================================================================
elif page == "🔬 Pipeline Architecture":
    st.title("System Architecture & Algorithmic Process")
    st.caption("How the system ingests raw academic literature across formats, clusters domains, and extracts research gaps.")

    render_glowing_pipeline_component()

# ==============================================================================
# 19. ROUTING: ❓ FAQ & INTERACTIVE RESEARCH Q&A
# ==============================================================================
elif page == "❓ FAQ & Research Q&A":
    st.title("Frequently Asked Questions & Research Q&A")
    st.caption("Ask questions about your uploaded literature corpus across any domain.")

    with st.container(border=True):
        st.subheader("🔍 Ask a Question About Your Uploaded Papers")
        q_input = st.text_input("Your Question / Query", placeholder="e.g., What are the main limitations in biological protein dynamics or quantum decoherence?")

        if q_input:
            if not st.session_state.papers:
                st.warning("Please upload papers or load the multi-domain corpus to search.")
            else:
                keywords = [w.lower().strip("?,.!") for w in q_input.split() if len(w) > 3]
                matches = []
                for doc_name, meta in st.session_state.papers.items():
                    for sent in meta.get("sentences", []):
                        sent_low = sent.lower()
                        score = sum(1 for kw in keywords if kw in sent_low)
                        if score > 0:
                            matches.append((doc_name, sent, score))

                matches.sort(key=lambda x: x[2], reverse=True)

                if matches:
                    st.success(f"Found {len(matches)} relevant sentence matches across your literature:")
                    for doc_name, sent, sc in matches[:5]:
                        st.markdown(f"""
                        <div class="kpi-card" style="margin-bottom:10px; border-width: 1.5px;">
                            <div style="font-size:0.75rem; color:#ffaa00; font-weight:800;">SOURCE: {doc_name} (Relevance Score: {sc})</div>
                            <div style="font-size:0.88rem; color:{text_color}; margin-top:4px; line-height:1.5;">"{sent}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No exact matching sentences found. Try using broader technical keywords.")

# ==============================================================================
# 20. ROUTING: ⚙ SETTINGS & PLATFORM
# ==============================================================================
elif page == "⚙ Settings & Platform":
    st.title("Platform Details & Execution Environment")
    st.caption("Multi-domain engine capabilities and dependency health status.")

    with st.container(border=True):
        st.subheader("Platform Metadata")
        st.markdown("""
        - **Platform Name**: ResearchLens AI
        - **Supported Ingestion**: PDF, DOCX, TXT, MD, CSV, JSON, HTML
        - **Clustering**: Universal Cross-Disciplinary (AI, ML, CS, Biology, Medicine, Physics, Chemistry)
        - **Gap Discovery**: Adaptive Self-Learning (Continuous Word2Vec Induction + 1D-CNN Sequence Modeling)
        - **Release Edition**: @ 2026 Enterprise Academic Intelligence
        """)

    with st.container(border=True):
        st.subheader("System Dependencies Status")
        dep_cols = st.columns(4)
        with dep_cols[0]:
            st.success("PyMuPDF: ✓ Active")
        with dep_cols[1]:
            st.success("Scikit-Learn: ✓ Active")
        with dep_cols[2]:
            st.success("NLTK: ✓ Active")
        with dep_cols[3]:
            st.success("ReportLab: ✓ Active")


## python src/train_cnn.py : To train cnn model
## python -m streamlit run app.py : To run the app