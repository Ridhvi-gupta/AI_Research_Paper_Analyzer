try:
    import fitz
except ImportError:
    import pymupdf as fitz

def extract_pdf_info(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract full text and page count in a single pass."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
        return "\n".join(pages), len(doc)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    return extract_pdf_info(pdf_bytes)[0]

def get_page_count(pdf_bytes: bytes) -> int:
    return extract_pdf_info(pdf_bytes)[1]