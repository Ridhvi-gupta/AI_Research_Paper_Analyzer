import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_pdf_report(validated_gaps: dict, papers_count: int) -> io.BytesIO:
    """
    Compiles verified research gaps, citations, and actionable suggestions
    into a structured publication-grade A4 PDF Dossier.
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header Banner
    p.setFillColor(colors.HexColor("#080c14"))
    p.rect(0, height - 90, width, 90, fill=True, stroke=False)
    
    p.setFillColor(colors.HexColor("#ffaa00"))
    p.setFont("Helvetica-Bold", 18)
    p.drawString(45, height - 42, "RESEARCHLENS AI - EXECUTIVE GAP DOSSIER")
    
    p.setFillColor(colors.white)
    p.setFont("Helvetica", 9)
    p.drawString(45, height - 60, f"Generated automatically on {time.strftime('%Y-%m-%d %H:%M')} | Total Papers Analyzed: {papers_count}")
    p.drawString(45, height - 74, "Course: CSET214 Data Analysis Using Python | Research Intelligence Report")

    y = height - 120
    p.setFillColor(colors.HexColor("#0f172a"))
    p.setFont("Helvetica-Bold", 13)
    p.drawString(45, y, "Verified Research Gaps & Recommended Solutions:")
    y -= 25

    if not validated_gaps:
        p.setFont("Helvetica-Oblique", 10)
        p.drawString(45, y, "No research gaps were validated by the researcher.")
    else:
        for idx, (gid, g) in enumerate(validated_gaps.items(), 1):
            if y < 110:
                p.showPage()
                y = height - 60

            # Source Paper & Score
            p.setFillColor(colors.HexColor("#b45309"))
            p.setFont("Helvetica-Bold", 10)
            p.drawString(45, y, f"[{idx}] Source Paper: {g['paper']}")
            
            p.setFillColor(colors.HexColor("#475569"))
            p.setFont("Helvetica-Bold", 9)
            p.drawRightString(width - 45, y, f"Hybrid Score: {g['hybrid_score']} | CNN: {g['cnn_prob']*100:.1f}%")
            y -= 16

            # Mined Sentence
            p.setFillColor(colors.black)
            p.setFont("Helvetica-Oblique", 9)
            sent_text = g['sentence']
            # Wrap long sentences
            if len(sent_text) > 105:
                p.drawString(55, y, f"\"{sent_text[:100]}...")
                y -= 13
                p.drawString(60, y, f"{sent_text[100:200]}\"")
            else:
                p.drawString(55, y, f"\"{sent_text}\"")
            y -= 15

            # Actionable Suggestion Box
            p.setFillColor(colors.HexColor("#065f46"))
            p.setFont("Helvetica-Bold", 8.5)
            s_type = g.get('suggestion_type', 'Research Recommendation')
            s_text = g.get('suggestion_text', 'Further empirical benchmarks needed.')
            p.drawString(55, y, f"Actionable Direction ({s_type}):")
            y -= 13
            p.setFont("Helvetica", 8)
            p.drawString(60, y, f"{s_text[:115]}")
            y -= 24

            # Subtle Divider Line
            p.setStrokeColor(colors.HexColor("#e2e8f0"))
            p.line(45, y + 8, width - 45, y + 8)

    p.save()
    buffer.seek(0)
    return buffer