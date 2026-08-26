"""
Resume Builder
----------------
Generates a clean, ATS-friendly resume as a .docx file.

ATS-friendly by construction, not by luck:
  - Single column, no tables/text boxes/images (the #1 cause of ATS parsing
    failures - many ATS systems literally cannot read text inside a table
    or text box, and silently drop it)
  - Standard section headers matching what our own ats_scorer.py looks for
    (Skills, Experience, Education, Projects, Certifications, Contact Info)
  - Plain bullet points (python-docx's built-in 'List Bullet' style, not
    manual dashes/symbols, which some ATS parsers mis-read)
  - No headers/footers (some ATS systems skip header/footer text entirely)
"""

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def build_resume_docx(data, output_path):
    """data is a dict with keys: name, email, phone, linkedin, summary, skills
    (list), experience (list of dicts: title, company, dates, bullets),
    education (list of dicts: degree, institution, year, score), certifications
    (list), projects (list of dicts: name, bullets).
    """
    doc = Document()

    # Base font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # --- Header: name + contact line ---
    name_p = doc.add_paragraph()
    name_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = name_p.add_run(data.get("name", "Your Name"))
    run.bold = True
    run.font.size = Pt(20)

    contact_bits = [b for b in [data.get("email"), data.get("phone"), data.get("linkedin")] if b]
    if contact_bits:
        contact_p = doc.add_paragraph(" | ".join(contact_bits))
        contact_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_section_heading(text):
        p = doc.add_paragraph()
        run = p.add_run(text.upper())
        run.bold = True
        run.font.size = Pt(13)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    # --- Summary ---
    if data.get("summary"):
        add_section_heading("Professional Summary")
        doc.add_paragraph(data["summary"])

    # --- Skills ---
    if data.get("skills"):
        add_section_heading("Skills")
        doc.add_paragraph(", ".join(data["skills"]))

    # --- Experience ---
    if data.get("experience"):
        add_section_heading("Experience")
        for exp in data["experience"]:
            line = doc.add_paragraph()
            r1 = line.add_run(f"{exp.get('title','')} — {exp.get('company','')}")
            r1.bold = True
            if exp.get("dates"):
                r2 = line.add_run(f"  ({exp['dates']})")
                r2.italic = True
            for bullet in exp.get("bullets", []):
                if bullet.strip():
                    doc.add_paragraph(bullet.strip(), style="List Bullet")

    # --- Projects ---
    if data.get("projects"):
        add_section_heading("Projects")
        for proj in data["projects"]:
            line = doc.add_paragraph()
            r1 = line.add_run(proj.get("name", ""))
            r1.bold = True
            for bullet in proj.get("bullets", []):
                if bullet.strip():
                    doc.add_paragraph(bullet.strip(), style="List Bullet")

    # --- Education ---
    if data.get("education"):
        add_section_heading("Education")
        for edu in data["education"]:
            parts = [edu.get("degree", ""), edu.get("institution", "")]
            line = " — ".join(p for p in parts if p)
            if edu.get("year"):
                line += f" ({edu['year']})"
            if edu.get("score"):
                line += f" | {edu['score']}"
            doc.add_paragraph(line)

    # --- Certifications ---
    if data.get("certifications"):
        add_section_heading("Certifications")
        for cert in data["certifications"]:
            if cert.strip():
                doc.add_paragraph(cert.strip(), style="List Bullet")

    doc.save(output_path)
    return output_path
