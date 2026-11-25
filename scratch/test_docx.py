import zipfile
import io
import re
import html
import xml.etree.ElementTree as ET

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Robust zero-dependency extractor for all DOCX variations, tables, and namespaces."""
    paragraphs = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
            # Gather all potential word XML streams
            namelist = docx.namelist()
            doc_xmls = [f for f in namelist if f.startswith('word/') and f.endswith('.xml')]
            # Prioritize document.xml, then document2.xml, then tables/headers
            doc_xmls.sort(key=lambda x: (0 if 'document' in x else 1, x))

            for xml_file in doc_xmls:
                if not any(k in xml_file for k in ['document', 'header', 'footer', 'notes']):
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

            # Secondary regex fallback if ET was empty or namespaces were non-standard

            if not paragraphs:
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

# Test creating a real mock docx in memory and extracting it
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w') as z:
    z.writestr('word/document.xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
  <w:p><w:r><w:t>Paper 1 Abstract: Machine learning needs robust evaluation.</w:t></w:r></w:p>
  <w:tbl><w:tr><w:tc><w:p><w:r><w:t xml:space="preserve">Paper 2 Abstract: In clinical trials, </w:t><w:t>sample size is severely limited.</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body>
</w:document>''')

extracted = extract_text_from_docx_bytes(buf.getvalue())
print("EXTRACTED RESULT:")
print(extracted)
print("WORD COUNT:", len(extracted.split()))
