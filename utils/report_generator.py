"""
ReportLab PDF Report Generation Module.
Compiles patient clinical parameters, model prediction, calibrated risk score,
explainability notes, and safety disclaimers into a formal PDF report.
"""

import io
from typing import Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from utils.preprocessing import FEATURE_METADATA


def generate_pdf_report(prediction_result: Dict[str, Any]) -> bytes:
    """
    Builds a professional clinical demonstration PDF document and returns the PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0369a1"),
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )

    normal_text = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#b91c1c"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("CARDIO GUARD CLINICAL ANALYTICS", title_style))
    story.append(
        Paragraph(
            f"Automated Risk Assessment Report &middot; Generated: {prediction_result.get('timestamp', 'N/A')}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=15))

    # 2. Executive Result Callout Box
    is_positive = prediction_result.get("prediction") == 1
    badge_bg = colors.HexColor("#fee2e2") if is_positive else colors.HexColor("#dcfce7")
    badge_border = colors.HexColor("#dc2626") if is_positive else colors.HexColor("#16a34a")
    pred_text_color = "#991b1b" if is_positive else "#166534"

    res_box_data = [
        [
            Paragraph(
                f"<b>PREDICTED CLASSIFICATION:</b> <font color='{pred_text_color}'><b>{prediction_result.get('prediction_label', 'N/A').upper()}</b></font>",
                normal_text,
            ),
            Paragraph(
                f"<b>ESTIMATED RISK TIER:</b> <font color='{pred_text_color}'><b>{prediction_result.get('risk_level', 'N/A').upper()}</b></font>",
                normal_text,
            ),
        ],
        [
            Paragraph(
                f"<b>Disease Probability:</b> {prediction_result.get('probability_disease', 0.0) * 100:.2f}%",
                normal_text,
            ),
            Paragraph(
                f"<b>Baseline Healthy Probability:</b> {prediction_result.get('probability_no_disease', 0.0) * 100:.2f}%",
                normal_text,
            ),
        ],
    ]
    res_table = Table(res_box_data, colWidths=[260, 270])
    res_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
                ("BOX", (0, 0), (-1, -1), 1.5, badge_border),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(res_table)
    story.append(Spacer(1, 15))

    # 3. Model Explainability & Summary
    story.append(Paragraph("Diagnostic Context & Model Attribution", section_heading))
    shap_info = prediction_result.get("shap_analysis", {})
    summary_text = shap_info.get("summary", "Inference computed via ensemble voting across 100 decision trees.")
    story.append(
        Paragraph(
            f"<b>Primary Decision Influencers:</b> {summary_text}",
            normal_text,
        )
    )
    story.append(Spacer(1, 10))

    # 4. Input Clinical Parameters Table
    story.append(Paragraph("Patient Recorded Parameters (13 Clinical Markers)", section_heading))
    inputs = prediction_result.get("input_parameters", {})

    table_data = [
        [
            Paragraph("<b>Parameter</b>", normal_text),
            Paragraph("<b>Recorded Value</b>", normal_text),
            Paragraph("<b>Parameter</b>", normal_text),
            Paragraph("<b>Recorded Value</b>", normal_text),
        ]
    ]

    keys = list(inputs.keys())
    for i in range(0, len(keys), 2):
        k1 = keys[i]
        meta1 = FEATURE_METADATA.get(k1, {})
        v1 = inputs[k1]
        label1 = meta1.get("label", k1)
        if "options" in meta1 and v1 in meta1["options"]:
            val_str1 = meta1["options"][v1]
        else:
            val_str1 = f"{v1} {meta1.get('unit', '')}".strip()

        if i + 1 < len(keys):
            k2 = keys[i + 1]
            meta2 = FEATURE_METADATA.get(k2, {})
            v2 = inputs[k2]
            label2 = meta2.get("label", k2)
            if "options" in meta2 and v2 in meta2["options"]:
                val_str2 = meta2["options"][v2]
            else:
                val_str2 = f"{v2} {meta2.get('unit', '')}".strip()
        else:
            label2, val_str2 = "", ""

        table_data.append(
            [
                Paragraph(label1, normal_text),
                Paragraph(val_str1, normal_text),
                Paragraph(label2, normal_text),
                Paragraph(val_str2, normal_text),
            ]
        )

    param_table = Table(table_data, colWidths=[140, 125, 140, 125])
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(param_table)
    story.append(Spacer(1, 15))

    # 5. Technical Pipeline Specifications
    story.append(Paragraph("System Provenance & Pipeline Specifications", section_heading))
    story.append(
        Paragraph(
            "<b>Algorithm:</b> Random Forest Classifier &middot; <b>Preprocessing:</b> Median/Mode Imputation, StandardScaler, OneHotEncoder &middot; "
            "<b>Dataset:</b> UCI Cleveland Heart Disease Database (ID 45) &middot; <b>Validation:</b> Stratified 80/20 Holdout",
            normal_text,
        )
    )
    story.append(Spacer(1, 12))

    # 6. Prominent Medical Safety Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=10))
    story.append(
        Paragraph(
            "<b>IMPORTANT CLINICAL & LEGAL DISCLAIMER:</b><br/>"
            "This document is an automated output generated by an educational Machine Learning demonstration project. "
            "<b>THIS IS NOT A CLINICALLY CERTIFIED MEDICAL DIAGNOSIS.</b> The predictive probabilities reflect statistical pattern recognition "
            "on historical research datasets and cannot substitute for coronary angiography, diagnostic laboratory testing, or consultation "
            "with a licensed cardiologist or physician. Never alter medications, treatments, or clinical protocols based on this output.",
            disclaimer_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
