"""
Generates synthetic medical document images using reportlab + Pillow.
Creates 3 document types: Prescription, Lab Report, Discharge Summary.
Run this FIRST to create test images.
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image
import subprocess

OUTPUT_DIR = Path("data/raw_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()


# ── Document 1: Prescription ──────────────────────────────────────────────────
def generate_prescription(filename: str):
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / filename),
        pagesize=letter,
        topMargin=0.5*inch
    )
    story = []

    title_style = ParagraphStyle(
        "title", fontSize=16, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    story.append(Paragraph("MEDICAL PRESCRIPTION", title_style))
    story.append(Spacer(1, 0.2*inch))

    header_data = [
        ["Hospital:", "UNC Medical Center"],
        ["Doctor:",   "Dr. Emily Johnson, MD"],
        ["License:",  "NC-MED-12345"],
        ["Date:",     "02/20/2024"],
    ]
    header_table = Table(header_data, colWidths=[2*inch, 4*inch])
    header_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 11),
        ("FONTNAME",  (0, 0), (0, -1),  "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.2*inch))

    patient_data = [
        ["Patient Name:", "John Smith"],
        ["DOB:",          "03/12/1959"],
        ["MRN:",          "1234567"],
        ["Allergies:",    "Penicillin, Sulfa"],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
        ("BACKGROUND",  (0, 0), (-1, -1), colors.lightgrey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Rx — Medications Prescribed:", styles["Heading2"]))
    story.append(Spacer(1, 0.1*inch))

    rx_data = [
        ["#", "Drug Name",       "Dosage",  "Frequency",  "Duration"],
        ["1", "Metformin",       "500mg",   "Twice daily","90 days"],
        ["2", "Lisinopril",      "10mg",    "Once daily", "90 days"],
        ["3", "Atorvastatin",    "40mg",    "Once nightly","90 days"],
        ["4", "Aspirin",         "81mg",    "Once daily", "Ongoing"],
        ["5", "Metoprolol",      "25mg",    "Twice daily","90 days"],
    ]
    rx_table = Table(rx_data, colWidths=[0.4*inch, 1.8*inch, 1*inch, 1.5*inch, 1.3*inch])
    rx_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.darkblue),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightblue]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rx_table)
    story.append(Spacer(1, 0.4*inch))

    story.append(Paragraph("Doctor Signature: ______________________", styles["Normal"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Refills: 2    DEA#: AB1234567", styles["Normal"]))

    doc.build(story)
    print(f"✅ Generated: {filename}")


# ── Document 2: Lab Report ────────────────────────────────────────────────────
def generate_lab_report(filename: str):
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / filename),
        pagesize=letter,
        topMargin=0.5*inch
    )
    story = []

    title_style = ParagraphStyle(
        "title", fontSize=16, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    story.append(Paragraph("LABORATORY REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))

    info_data = [
        ["Patient:",   "Maria Garcia",     "Accession #:", "LAB-20240220-001"],
        ["DOB:",       "11/05/1984",        "Collected:",   "02/20/2024 09:15"],
        ["MRN:",       "9876543",           "Reported:",    "02/20/2024 14:30"],
        ["Physician:", "Dr. Robert Chen",   "Lab:",         "Duke Clinical Lab"],
    ]
    info_table = Table(info_data, colWidths=[1.2*inch, 1.8*inch, 1.3*inch, 2.2*inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1),  "Helvetica-Bold"),
        ("BACKGROUND",  (0, 0), (-1, -1), colors.lightyellow),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("COMPREHENSIVE METABOLIC PANEL", styles["Heading2"]))
    story.append(Spacer(1, 0.1*inch))

    lab_data = [
        ["Test",          "Result", "Units",   "Reference Range", "Flag"],
        ["Glucose",       "285",    "mg/dL",   "70-99",           "H ⚠"],
        ["BUN",           "18",     "mg/dL",   "7-20",            ""],
        ["Creatinine",    "0.9",    "mg/dL",   "0.6-1.2",         ""],
        ["Sodium",        "138",    "mEq/L",   "136-145",         ""],
        ["Potassium",     "3.8",    "mEq/L",   "3.5-5.0",         ""],
        ["HbA1c",         "9.2",    "%",        "<5.7",            "H ⚠"],
        ["Total Chol.",   "245",    "mg/dL",   "<200",            "H"],
        ["HDL",           "38",     "mg/dL",   ">40",             "L"],
        ["LDL",           "162",    "mg/dL",   "<100",            "H ⚠"],
        ["Triglycerides", "225",    "mg/dL",   "<150",            "H"],
    ]
    lab_table = Table(
        lab_data,
        colWidths=[2*inch, 0.8*inch, 0.8*inch, 1.8*inch, 0.8*inch]
    )
    lab_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.darkblue),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("TEXTCOLOR",     (4, 1), (4, -1),  colors.red),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(lab_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "⚠ Critical/Abnormal values flagged. Physician review required.",
        styles["Normal"]
    ))

    doc.build(story)
    print(f"✅ Generated: {filename}")


# ── Document 3: Discharge Summary ────────────────────────────────────────────
def generate_discharge_summary(filename: str):
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / filename),
        pagesize=letter,
        topMargin=0.5*inch
    )
    story = []

    title_style = ParagraphStyle(
        "title", fontSize=16, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    story.append(Paragraph("DISCHARGE SUMMARY", title_style))
    story.append(Spacer(1, 0.2*inch))

    info_data = [
        ["Patient:",    "David Lee",          "MRN:",         "A123-45-6789"],
        ["DOB:",        "07/21/1972",          "Admit Date:",  "01/15/2024"],
        ["Physician:",  "Dr. Sarah Williams",  "Discharge:",   "01/18/2024"],
        ["Hospital:",   "Atrium Health Charlotte", "LOS:",     "3 days"],
    ]
    info_table = Table(info_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 1.8*inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1),  "Helvetica-Bold"),
        ("BACKGROUND",  (0, 0), (-1, -1), colors.lightblue),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2*inch))

    sections = [
        ("Primary Diagnosis",
         "Acute exacerbation of COPD with secondary community-acquired pneumonia. "
         "Patient presented with worsening dyspnea, productive cough, and O2 sat 88%."),
        ("Secondary Diagnoses",
         "1. Hypertension (controlled)\n2. Type 2 Diabetes Mellitus\n3. Hyperlipidemia"),
        ("Hospital Course",
         "Patient admitted via ED on 01/15/2024. Started on IV Azithromycin and "
         "Ceftriaxone. Nebulizer therapy initiated. O2 supplementation weaned over "
         "48 hours. Stable on room air by Day 3."),
        ("Discharge Medications",
         "1. Azithromycin 250mg orally once daily x 3 days\n"
         "2. Prednisone 40mg orally once daily x 5 days\n"
         "3. Albuterol inhaler 2 puffs every 4-6 hours PRN\n"
         "4. Metformin 1000mg twice daily (continue)\n"
         "5. Lisinopril 10mg once daily (continue)"),
        ("Follow-Up Instructions",
         "Follow up with PCP Dr. Sarah Williams within 7 days (02/01/2024). "
         "Pulmonology referral placed. Return to ED if dyspnea worsens or "
         "O2 sat drops below 92%."),
        ("Discharge Condition",
         "Stable. Afebrile. O2 sat 96% on room air. Ambulating independently."),
    ]

    for heading, content in sections:
        story.append(Paragraph(heading, styles["Heading3"]))
        story.append(Paragraph(content.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 0.15*inch))

    doc.build(story)
    print(f"✅ Generated: {filename}")


if __name__ == "__main__":
    print("📄 Generating synthetic medical documents...\n")
    generate_prescription("prescription_001.pdf")
    generate_lab_report("lab_report_001.pdf")
    generate_discharge_summary("discharge_summary_001.pdf")
    print("\n✅ All documents generated in data/raw_images/")
