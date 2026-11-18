"""Generate downloadable, self-contained PDF reports for demo predictions."""

from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.preprocessing import FEATURE_METADATA, FEATURE_NAMES


def generate_prediction_pdf(
    input_values: Dict[str, Any],
    result: Dict[str, Any],
    created_at: datetime,
    explanation: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """Create a PDF containing the submitted profile, model score, and disclaimer."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="CardioGuard educational prediction report",
        author="CardioGuard educational project",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            textColor=colors.HexColor("#647789"),
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSection",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#172B3A"),
            fontSize=13,
            leading=17,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportDisclaimer",
            parent=styles["Normal"],
            textColor=colors.HexColor("#704C3D"),
            fontSize=8.5,
            leading=12,
        )
    )

    story = [
        Paragraph("CardioGuard", styles["Title"]),
        Paragraph("Educational heart disease model report", styles["ReportSubtitle"]),
        Spacer(1, 8),
        Paragraph(
            "Prediction timestamp: "
            + escape(
                created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ),
            styles["Normal"],
        ),
        Paragraph("Submitted profile", styles["ReportSection"]),
    ]

    profile_rows = [["Clinical feature", "Submitted value"]]
    for feature in FEATURE_NAMES:
        metadata = FEATURE_METADATA[feature]
        value = input_values[feature]
        unit = metadata.get("unit")
        rendered_value = f"{value} {unit}" if unit else str(value)
        profile_rows.append([metadata["label"], rendered_value])

    profile_table = Table(profile_rows, colWidths=[3.5 * inch, 3.2 * inch], repeatRows=1)
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173746")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E1E8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    score = float(result["probability_disease"]) * 100
    output_label = (
        "Model output: class 1"
        if int(result["prediction"]) == 1
        else "Model output: class 0"
    )
    explanation_rows = explanation or []
    explanation_text = (
        "<br/>".join(
            (
                f"{escape(str(row['Feature']))}: changing this value to the cohort "
                f"reference changed the model score by "
                f"{float(row['Score change']) * 100:+.1f} percentage points."
            )
            for row in explanation_rows[:3]
        )
        if explanation_rows
        else "A local feature sensitivity explanation was not available for this report."
    )
    story.extend(
        [
            profile_table,
            Paragraph("Model output", styles["ReportSection"]),
            Paragraph(f"<b>{output_label}</b>", styles["Normal"]),
            Paragraph(
                f"Estimated model score for class 1: {score:.1f}% "
                "(not calibrated for clinical risk).",
                styles["Normal"],
            ),
            Paragraph("Explanation", styles["ReportSection"]),
            Paragraph(
                "The output is produced by a Random Forest trained on the UCI "
                "Cleveland Heart Disease dataset. " + explanation_text + " "
                "This one-feature-at-a-time sensitivity view is descriptive, not "
                "causal, and does not account for feature interactions.",
                styles["Normal"],
            ),
            Paragraph("Medical disclaimer", styles["ReportSection"]),
            Paragraph(
                "<b>Educational demonstration only.</b> This software is not a "
                "clinically validated medical device and must not be used for "
                "self-diagnosis, clinical decision-making, or treatment. Consult "
                "a licensed healthcare professional for medical concerns.",
                styles["ReportDisclaimer"],
            ),
        ]
    )
    document.build(story)
    return output.getvalue()
