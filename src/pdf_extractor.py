import re
import pymupdf
from src.preprocessing import segment_sentences

def extract_pdf_info(pdf_path_or_bytes) -> dict:
    """
    Extracts text and metadata from PDF using single-pass PyMuPDF.
    Automatically cuts off References/Bibliography to eliminate citation noise.
    """
    if isinstance(pdf_path_or_bytes, bytes):
        doc = pymupdf.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path_or_bytes)
        
    page_count = len(doc)
    full_text = ""
    pages_text = []

    for page_idx, page in enumerate(doc):
        text = page.get_text() or ""
        pages_text.append(text)
        full_text += text + "\n"

    # Section-Aware Cutoff: Remove Bibliography / References to prevent false positive gaps
    # Search for References header near the end of the paper
    ref_match = re.search(r'\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n', full_text)
    if ref_match:
        analysis_text = full_text[:ref_match.start()]
    else:
        analysis_text = full_text

    # Extract sentences using academic noise cleaner
    sentences = segment_sentences(analysis_text)
    words = analysis_text.split()

    return {
        "text": analysis_text,
        "full_text_with_refs": full_text,
        "page_count": page_count,
        "word_count": len(words),
        "char_count": len(analysis_text),
        "sentences": sentences
    }