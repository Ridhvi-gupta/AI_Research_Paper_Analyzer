import io
import os
import json
import zipfile
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_modified_document_content(original_text: str, doc_name: str, validated_gaps: list[dict]) -> str:
    """
    Appends a formal, auditable 'Validated Research Gaps & Strategic Amendments'
    section to the document text, resolving limitations with actionable proposals.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    amendment_lines = [
        "",
        "=" * 80,
        "SECTION: VALIDATED RESEARCH GAPS & METHODOLOGICAL AMENDMENTS",
        f"Document: {doc_name}",
        f"Audited & Validated on: {now_str}",
        f"Total Research Gaps Addressed: {len(validated_gaps)}",
        "=" * 80,
        ""
    ]

    for idx, gap in enumerate(validated_gaps, 1):
        sent = gap.get("sentence", "").strip()
        s_type = gap.get("suggestion_type", "Methodological Improvement")
        s_text = gap.get("suggestion_text", "Further investigation required.")
        rules = gap.get("rules", [])
        hybrid_sc = gap.get("hybrid_score", 0.0)
        cnn_sc = gap.get("cnn_prob", 0.0)

        trigger_info = ", ".join(rules) if rules else "CNN Neural Sequence Bottleneck"

        amendment_lines.extend([
            f"[AMENDMENT #{idx}]",
            f"• Original Limitation Claim: \"{sent}\"",
            f"• Deficit Classification:   {s_type}",
            f"• Trigger Attribution:      {trigger_info}",
            f"• Confidence Metrics:       Hybrid: {hybrid_sc} | CNN Structural Probability: {cnn_sc*100:.1f}%",
            f"• Strategic Resolution:     {s_text}",
            "-" * 80,
            ""
        ])

    return original_text.rstrip() + "\n\n" + "\n".join(amendment_lines)

def generate_modified_pdf(doc_name: str, original_text: str, validated_gaps: list[dict]) -> io.BytesIO:
    """Generates an A4/Letter PDF of the modified document with styled callouts."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )

    section_hdr = ParagraphStyle(
        'SectionHdr',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#ea580c'),
        spaceBefore=16,
        spaceAfter=8
    )

    box_text = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0f172a')
    )

    elements = []

    # Title & Metadata
    clean_name = os.path.splitext(doc_name)[0]
    elements.append(Paragraph(f"{clean_name} (Enriched Revision)", title_style))
    elements.append(Paragraph(f"Audited with ResearchLens AI | Amendments Incorporated: {len(validated_gaps)} | Date: {datetime.now().strftime('%B %d, %Y')}", meta_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#ea580c'), spaceAfter=14))

    # Validated Gaps Section (Front & Center)
    if validated_gaps:
        elements.append(Paragraph("Formal Amendments & Addressed Research Gaps", section_hdr))
        for idx, gap in enumerate(validated_gaps, 1):
            sent = gap.get("sentence", "").strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            s_type = gap.get("suggestion_type", "Methodological Improvement")
            s_text = gap.get("suggestion_text", "")

            box_content = f"""
            <b>Amendment #{idx}: {s_type}</b><br/>
            <i>Original Limitation:</i> "{sent}"<br/><br/>
            <b>Proposed Resolution / Enhancement:</b> {s_text}
            """
            p = Paragraph(box_content, box_text)
            t = Table([[p]], colWidths=[520])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')),
                ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#f59e0b')),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 10))

    # Original Document Body
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Primary Document Content", section_hdr))
    paragraphs = original_text.split('\n\n')
    for p_text in paragraphs[:35]:
        clean_p = p_text.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if clean_p:
            elements.append(Paragraph(clean_p, body_style))

    doc.build(elements)
    buf.seek(0)
    return buf

def generate_modified_docx(doc_name: str, full_content: str) -> io.BytesIO:
    """Generates a valid OpenXML Word Document (.docx) using pure Python zipfile."""
    buf = io.BytesIO()
    clean_title = os.path.splitext(doc_name)[0]

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as docx:
        # 1. Content Types
        docx.writestr('[Content_Types].xml', (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="xml" ContentType="application/xml"/>\n'
            '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
            '</Types>'
        ))

        # 2. Package Relationships
        docx.writestr('_rels/.rels', (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
            '</Relationships>'
        ))

        # 3. Document Body
        p_xml_list = []
        p_xml_list.append(
            f'<w:p><w:pPr><w:jc w:val="left"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="EA580C"/></w:rPr>'
            f'<w:t>{clean_title} (Amended Revision)</w:t></w:r></w:p>'
        )

        for line in full_content.split('\n'):
            line_esc = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if line_esc.startswith('SECTION:') or line_esc.startswith('='):
                p_xml_list.append(
                    f'<w:p><w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="EA580C"/></w:rPr>'
                    f'<w:t>{line_esc}</w:t></w:r></w:p>'
                )
            elif line_esc.startswith('[AMENDMENT'):
                p_xml_list.append(
                    f'<w:p><w:r><w:rPr><w:b/><w:color w:val="059669"/></w:rPr>'
                    f'<w:t>{line_esc}</w:t></w:r></w:p>'
                )
            else:
                p_xml_list.append(f'<w:p><w:r><w:t>{line_esc}</w:t></w:r></w:p>')

        body_inner = "".join(p_xml_list)
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            f'  <w:body>{body_inner}</w:body>\n'
            '</w:document>'
        )
        docx.writestr('word/document.xml', doc_xml)

    buf.seek(0)
    return buf

def generate_modified_html(doc_name: str, original_text: str, validated_gaps: list[dict]) -> str:
    """Generates a responsive HTML report of the modified document with CSS styling."""
    clean_title = os.path.splitext(doc_name)[0]
    gaps_html = ""
    for idx, g in enumerate(validated_gaps, 1):
        gaps_html += f"""
        <div style="background:#fef3c7; border-left:4px solid #f59e0b; padding:12px 16px; margin-bottom:12px; border-radius:6px; color:#1e293b;">
            <div style="font-weight:bold; color:#b45309; font-size:1.05rem;">Amendment #{idx}: {g.get('suggestion_type', 'Enhancement')}</div>
            <p style="margin:6px 0; font-style:italic;"><strong>Identified Limitation:</strong> "{g.get('sentence', '')}"</p>
            <p style="margin:4px 0; color:#047857;"><strong>Strategic Actionable Resolution:</strong> {g.get('suggestion_text', '')}</p>
        </div>
        """

    paragraphs = "".join([f"<p style='line-height:1.6; margin-bottom:12px;'>{p.strip()}</p>" for p in original_text.split('\n\n') if p.strip()])

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{clean_title} - Enriched Document</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#0f172a; color:#f8fafc; max-width:900px; margin:40px auto; padding:0 20px; }}
        h1 {{ color:#ff7700; border-bottom:2px solid #ff7700; padding-bottom:10px; }}
        h2 {{ color:#ffe066; margin-top:30px; }}
        .meta {{ color:#94a3b8; font-size:0.9rem; margin-bottom:25px; }}
        .body-card {{ background:#1e293b; padding:25px; border-radius:10px; border:1px solid #334155; }}
    </style>
</head>
<body>
    <h1>{clean_title} (Amended Literature)</h1>
    <div class="meta">Audited with ResearchLens AI &bull; Total Validated Gaps Addressed: {len(validated_gaps)} &bull; {datetime.now().strftime('%Y-%m-%d')}</div>
    
    <h2>Formal Amendments & Strategic Resolutions</h2>
    {gaps_html if gaps_html else '<p>No research gaps currently attached.</p>'}

    <h2>Document Primary Content</h2>
    <div class="body-card">
        {paragraphs}
    </div>
</body>
</html>"""

def export_modified_file(doc_name: str, original_text: str, validated_gaps: list[dict], export_format: str) -> tuple[io.BytesIO, str, str]:
    """
    Main export dispatcher:
    Returns (bytes_buffer, filename, mime_type) based on user's selected format.
    Supports: PDF, DOCX, Markdown (MD), Plain Text (TXT), JSON, and HTML.
    """
    base_name = os.path.splitext(doc_name)[0]
    full_content = build_modified_document_content(original_text, doc_name, validated_gaps)

    # 1. PDF
    if export_format == "PDF (.pdf)":
        pdf_buf = generate_modified_pdf(doc_name, original_text, validated_gaps)
        return pdf_buf, f"{base_name}_Amended.pdf", "application/pdf"

    # 2. Word Document (DOCX)
    elif export_format == "Word Document (.docx)":
        docx_buf = generate_modified_docx(doc_name, full_content)
        return docx_buf, f"{base_name}_Amended.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # 3. Markdown (.md)
    elif export_format == "Markdown (.md)":
        md_text = f"# {base_name} (Amended Revision)\n\n"
        if validated_gaps:
            md_text += "## Validated Research Gaps & Strategic Resolutions\n\n"
            for idx, g in enumerate(validated_gaps, 1):
                md_text += f"### Amendment #{idx}: {g.get('suggestion_type', 'Enhancement')}\n"
                md_text += f"> **Limitation Claim:** \"{g.get('sentence', '')}\"\n>\n"
                md_text += f"> **Strategic Solution:** {g.get('suggestion_text', '')}\n\n"
        md_text += "## Primary Document Text\n\n" + original_text
        buf = io.BytesIO(md_text.encode('utf-8'))
        return buf, f"{base_name}_Amended.md", "text/markdown"

    # 4. Plain Text (.txt)
    elif export_format == "Plain Text (.txt)":
        buf = io.BytesIO(full_content.encode('utf-8'))
        return buf, f"{base_name}_Amended.txt", "text/plain"

    # 5. JSON (.json)
    elif export_format == "JSON (.json)":
        data = {
            "document_name": doc_name,
            "modified_at": datetime.now().isoformat(),
            "validated_gaps_count": len(validated_gaps),
            "validated_amendments": validated_gaps,
            "original_text": original_text,
            "modified_full_content": full_content
        }
        buf = io.BytesIO(json.dumps(data, indent=2).encode('utf-8'))
        return buf, f"{base_name}_Amended.json", "application/json"

    # 6. HTML (.html)
    elif export_format == "HTML (.html)":
        html_str = generate_modified_html(doc_name, original_text, validated_gaps)
        buf = io.BytesIO(html_str.encode('utf-8'))
        return buf, f"{base_name}_Amended.html", "text/html"

    # Default fallback to TXT
    buf = io.BytesIO(full_content.encode('utf-8'))
    return buf, f"{base_name}_Amended.txt", "text/plain"