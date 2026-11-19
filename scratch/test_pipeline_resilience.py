import sys, os
sys.path.insert(0, os.path.abspath('.'))
import io
import zipfile
from src.universal_extractor import extract_universal_document
from src.clustering import perform_universal_clustering
from src.document_exporter import export_modified_file, build_modified_document_content

print("--- 1. Testing Universal DOCX Extraction ---")
# Create mock DOCX with tables & paragraphs
docx_buf = io.BytesIO()
with zipfile.ZipFile(docx_buf, 'w') as z:
    z.writestr('word/document.xml', '''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
  <w:p><w:r><w:t>Deep learning models frequently suffer from catastrophic forgetting and lack of multimodal reasoning.</w:t></w:r></w:p>
  <w:tbl>
    <w:tr><w:tc><w:p><w:r><w:t>In addition, clinical validation remains an open challenge.</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
  <w:p><w:r><w:t>References</w:t></w:r></w:p>
  <w:p><w:r><w:t>[1] Devlin et al., 2019.</w:t></w:r></w:p>
</w:body>
</w:document>''')

doc_meta = extract_universal_document(docx_buf.getvalue(), "AllpaperAbstracts.docx")
print("Document Name:", doc_meta["filename"])
print("Word Count:", doc_meta["word_count"])
print("Text Preview:", doc_meta["text"][:100])
assert doc_meta["word_count"] > 0, "Word count should not be 0!"

print("\n--- 2. Testing Empty Text Handling in Clustering ---")
empty_papers = {
    "empty.txt": {
        "text": "   ",
        "word_count": 0,
        "filename": "empty.txt",
        "extension": "TXT"
    }
}
clust_res = perform_universal_clustering(empty_papers, k=1)
print("Clustering on empty corpus returned:", clust_res.get("error"))
assert "error" in clust_res

print("\n--- 3. Testing Single-Document Clustering ---")
single_paper = {
    "AllpaperAbstracts.docx": doc_meta
}
clust_single = perform_universal_clustering(single_paper, k=1)
print("Single-doc cluster count:", len(clust_single["df_clusters"]))
assert len(clust_single["df_clusters"]) == 1

print("\n--- 4. Testing Multi-Format Export ---")
sample_gaps = [
    {
        "sentence": "Deep learning models suffer from catastrophic forgetting.",
        "category": "Methodological & Theoretical Gaps",
        "hybrid_score": 0.92,
        "severity": "CRITICAL",
        "resolution": "Implement rehearsal buffers and elastic weight consolidation."
    }
]

format_labels = [
    "PDF (.pdf)",
    "Word Document (.docx)",
    "Plain Text (.txt)",
    "Markdown (.md)",
    "JSON (.json)",
    "HTML (.html)"
]

for fmt in format_labels:
    buf, dl_name, mime = export_modified_file(
        doc_name=doc_meta["filename"],
        original_text=doc_meta["text"],
        validated_gaps=sample_gaps,
        export_format=fmt
    )
    val = buf.getvalue()
    print(f"Export format {fmt:25s} -> {len(val):6d} bytes ({dl_name})")
    assert len(val) > 0

print("\nALL RESILIENCE TESTS PASSED SUCCESSFULLY!")
