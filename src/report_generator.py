import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(output_path: str, papers: list[dict], gap_results: list[dict]) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("<b>AI Research Paper Analyzer — Research Gap Report</b>", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"<b>Total Papers:</b> {len(papers)} | <b>Potential Gaps Mined:</b> {len(gap_results)}", styles["Normal"]),
        Spacer(1, 15),
        Paragraph("<b>Top Candidate Research Gap Sentences</b>", styles["Heading2"]),
        Spacer(1, 8),
    ]

    table_data = [["Paper", "Candidate Sentence", "CNN Score", "Score"]]
    for res in gap_results[:25]:
        sent = (res["sentence"][:120] + "...") if len(res["sentence"]) > 120 else res["sentence"]
        table_data.append([res.get("paper", "Unknown"), sent, str(res["cnn_score"]), str(res["combined_score"])])

    tbl = Table(table_data, colWidths=[100, 270, 75, 75])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5c8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    elements.append(tbl)
    doc.build(elements)
    return output_path