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
    """Robust zero-dependency extractor for all DOCX variations, tables, and namespaces."""
    paragraphs = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
            namelist = docx.namelist()
            doc_xmls = [f for f in namelist if f.startswith('word/') and f.endswith('.xml')]
            doc_xmls.sort(key=lambda x: (0 if 'document' in x else 1, x))

            for xml_file in doc_xmls:
                if not any(k in xml_file for k in ['document', 'header', 'footer', 'notes', 'table']):
                    continue
                try:
                    xml_content = docx.read(xml_file)
                    tree = ET.fromstring(xml_content)
                    for elem in tree.iter():
                        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        if tag in ['p', 'tr']:
                            p_texts = []
                            for child in elem.iter():
                                ctag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                                if ctag == 't' and child.text:
                                    p_texts.append(child.text)
                            line = ''.join(p_texts).strip()
                            if line and (not paragraphs or paragraphs[-1] != line):
                                paragraphs.append(line)
                except Exception:
                    pass

            if not paragraphs:
                import html
                for xml_file in doc_xmls:
                    if 'document' in xml_file:
                        raw_str = docx.read(xml_file).decode('utf-8', errors='ignore')
                        matches = re.findall(r'<[^:>]*:?t\b[^>]*>(.*?)</[^:>]*:?t>', raw_str, flags=re.DOTALL)
                        if matches:
                            cleaned = [html.unescape(m).strip() for m in matches if m.strip()]
                            if cleaned:
                                paragraphs.append(' '.join(cleaned))
    except Exception as e:
        return f"DOCX parsing fallback: {str(e)}"

    return '\n\n'.join(paragraphs)

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
            clean_html = re.sub(r'<script.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            clean_html = re.sub(r'<style.*?</style>', '', clean_html, flags=re.DOTALL | re.IGNORECASE)
            full_text = re.sub(r'<[^>]+>', ' ', clean_html)
            page_count = max(1, len(full_text.split()) // 450)

        else:
            full_text = file_bytes.decode("utf-8", errors="ignore")
            page_count = max(1, len(full_text.split()) // 450)

    except Exception as err:
        full_text = f"Error reading document: {err}"

    # Section-Aware Reference Cutoff
    ref_match = re.search(r'\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n', full_text)
    if ref_match and ref_match.start() > 250:
        candidate_text = full_text[:ref_match.start()].strip()
        if len(candidate_text.split()) >= 50:
            analysis_text = candidate_text
        else:
            analysis_text = full_text
    else:
        analysis_text = full_text

    # Guard against completely blank extractions
    if not analysis_text.strip():
        analysis_text = full_text.strip() if full_text.strip() else f"Document {filename} ingested."

    sentences = segment_sentences(analysis_text)
    words = analysis_text.split()

    return {
        "text": analysis_text,
        "filename": filename,
        "extension": ext.upper().replace(".", ""),
        "page_count": page_count,
        "word_count": max(1, len(words)),
        "char_count": len(analysis_text),
        "sentences": sentences
    }