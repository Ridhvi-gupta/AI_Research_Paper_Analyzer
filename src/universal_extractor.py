import os
import io
import re
import json
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import pymupdf
from src.preprocessing import segment_sentences

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Extracts text from DOCX using zipfile & XML without external docx dependency."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"DOCX parsing fallback: {str(e)}"

def extract_universal_document(file_bytes: bytes, filename: str) -> dict:
    """
    Universal Ingestion Engine:
    Parses PDF, DOCX, TXT, MD, CSV, JSON, and HTML documents.
    Strips bibliography/references and segments clean sentences.
    """
    ext = os.path.splitext(filename)[1].lower()
    full_text = ""
    page_count = 1

    try:
        # 1. PDF
        if ext == ".pdf":
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            page_count = len(doc)
            for page in doc:
                full_text += (page.get_text() or "") + "\n"

        # 2. DOCX (Word Document)
        elif ext == ".docx":
            full_text = extract_text_from_docx_bytes(file_bytes)
            page_count = max(1, len(full_text.split()) // 450)

        # 3. Plain Text / Markdown
        elif ext in [".txt", ".md"]:
            full_text = file_bytes.decode("utf-8", errors="ignore")
            page_count = max(1, len(full_text.split()) // 450)

        # 4. JSON (e.g. arXiv API or Semantic Scholar export)
        elif ext == ".json":
            data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            if isinstance(data, list):
                full_text = "\n\n".join([str(item.get("abstract", "") or item.get("text", "") or item) for item in data])
            elif isinstance(data, dict):
                full_text = str(data.get("body", "") or data.get("abstract", "") or data.get("text", "") or json.dumps(data, indent=2))
            page_count = max(1, len(full_text.split()) // 450)

        # 5. CSV / TSV (Tabular Paper Repositories)
        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
            text_cols = [c for c in df.columns if any(k in c.lower() for k in ["text", "abstract", "content", "summary", "paper"])]
            if text_cols:
                full_text = "\n\n".join(df[text_cols[0]].dropna().astype(str).tolist())
            else:
                full_text = "\n\n".join(df.astype(str).apply(" ".join, axis=1).tolist())
            page_count = max(1, len(df))

        # 6. HTML / HTM (Web research publications)
        elif ext in [".html", ".htm"]:
            raw_html = file_bytes.decode("utf-8", errors="ignore")
            # Strip script, style, and HTML tags
            clean_html = re.sub(r'<script.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            clean_html = re.sub(r'<style.*?</style>', '', clean_html, flags=re.DOTALL | re.IGNORECASE)
            full_text = re.sub(r'<[^>]+>', ' ', clean_html)
            page_count = max(1, len(full_text.split()) // 450)

        else:
            # General fallback decode
            full_text = file_bytes.decode("utf-8", errors="ignore")
            page_count = max(1, len(full_text.split()) // 450)

    except Exception as err:
        full_text = f"Error reading document: {err}"

    # Section-Aware Reference Cutoff
    ref_match = re.search(r'\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n', full_text)
    if ref_match:
        analysis_text = full_text[:ref_match.start()]
    else:
        analysis_text = full_text

    sentences = segment_sentences(analysis_text)
    words = analysis_text.split()

    return {
        "text": analysis_text,
        "filename": filename,
        "extension": ext.upper().replace(".", ""),
        "page_count": page_count,
        "word_count": len(words),
        "char_count": len(analysis_text),
        "sentences": sentences
    }