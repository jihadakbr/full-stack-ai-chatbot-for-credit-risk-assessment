from pathlib import Path  # top
import os
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Layout & spacing controls
SECTION_SPACING = 24  # vertical gap after each section
PARAGRAPH_LEADING = 14  # line spacing within paragraphs / tables
MARGIN = 72  # 1 inch margin

# Dev mode
# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Goes up 3 levels
# OUTPUT_DIR = PROJECT_ROOT / "generated_pdfs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Docker mode
OUTPUT_DIR = Path(os.getenv("GENERATED_PDFS_DIR", "/opt/airflow/generated_pdfs"))

FOOTER_TEXT = (
    "This report provides model-based explanations to aid review and is not a "
    "standalone credit decision."
)

EXPLANATION_PARAGRAPHS = [
    (
        "What the prediction means",
        "“Default” means the model predicts the applicant is likely to miss loan payments or fall seriously behind within the next 12 months. "
        "“Non-Default” means the model predicts the applicant is likely to make all loan payments on time.",
    ),
    (
        "Prediction confidence (probability)",
        "The percentage shown is the model’s confidence in the predicted class for this applicant at the time of scoring. "
        "Examples: Default (70%) means ~70% probability of default. Non-Default (80%) means ~80% probability of NOT defaulting "
        "(i.e., ~20% probability of default).",
    ),
    (
        "About the 20 feature contributions",
        "Below we list up to 20 model inputs (features) that most influenced this individual prediction, ranked by impact. "
        "Each feature has a contribution value from SHAP (a standard explainability method). The larger the absolute value, "
        "the bigger the effect that feature had on pushing the prediction toward Default or Non-Default.",
    ),
    (
        "How to read positive vs. negative values",
        "Positive SHAP values push the prediction toward Default (higher risk). "
        "Negative SHAP values push the prediction toward Non-Default (lower risk). "
        "Magnitude shows strength of influence, not goodness or badness on its own.",
    ),
    (
        "Important notes",
        "• This is a model-based estimate using the information available at the time of scoring.\n"
        "• Feature names reflect the data field used by the model. The contribution value is specific to this applicant and may differ for others.\n"
        "• Final lending decisions should consider internal policies, additional documentation, and relevant regulations.",
    ),
]

FEATURE_ORDER_20 = [
    "EXT_SOURCE_1",
    "GOODS_CREDIT_RATIO",
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "ORGANIZATION_TYPE_ENCODED",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "CODE_GENDER",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "FLAG_OWN_CAR",
    "DAYS_ID_PUBLISH",
    "ATI_RATIO",
    "OWN_CAR_AGE",
    "LIVINGAREA_MEDI",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_3",
    "NAME_FAMILY_STATUS_Married",
    "NAME_INCOME_TYPE_Working",
]

FEATURE_GLOSSARY = {
    "EXT_SOURCE_1": "Credit score from an external source #1. Higher values usually mean better creditworthiness.",
    "EXT_SOURCE_2": "Credit score from an external source #2. Higher values usually mean better creditworthiness.",
    "EXT_SOURCE_3": "Credit score from an external source #3. Higher values usually mean better creditworthiness.",
    "GOODS_CREDIT_RATIO": "The price of the goods compared to the loan amount. Higher values mean you pay more upfront (because the goods cost more than the loan). Lower values mean the loan covers most or more than the goods' price.",
    "ORGANIZATION_TYPE_ENCODED": "The type of company or organization you work for, represented as a code.",
    "DAYS_BIRTH": "Your age in days relative to the application date (stored as a negative number for prediction data, so a bigger negative number means older).",
    "DAYS_EMPLOYED": "How long you have been with your current employer relative to the application date, in days (a larger number means longer employment).",
    "NAME_EDUCATION_TYPE": "Your highest level of education (e.g., secondary, higher education, etc.).",
    "CODE_GENDER": "Your gender as recorded in the application.",
    "AMT_ANNUITY": "The regular payment amount (installment) you will make for the loan.",
    "AMT_CREDIT": "The total amount of the loan you requested.",
    "FLAG_OWN_CAR": "Shows whether you own a car (Yes or No).",
    "DAYS_ID_PUBLISH": "How many days ago your ID information was last updated or issued relative to the application date.",
    "ATI_RATIO": "Your income compared to the loan or payment amount. A higher value means the loan is more affordable for you.",
    "OWN_CAR_AGE": "The age of your car in years, if you own one.",
    "LIVINGAREA_MEDI": "A value showing the size of your living space, compared to others (normalized scale).",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Number of people in your social circle who had payments overdue by more than 30 days (based on credit bureau data).",
    "FLAG_DOCUMENT_3": "Shows whether you provided and verified a valid national ID.",
    "NAME_FAMILY_STATUS_Married": "Indicates if you are married.",
    "NAME_INCOME_TYPE_Working": "Indicates if your main source of income is from regular work.",
}


def _ensure_output_dir() -> Path:
    """
    Make sure OUTPUT_DIR exists and is writable. If it's not, fall back to /tmp.
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        test = OUTPUT_DIR / ".write_test"
        test.write_text("ok")
        test.unlink(missing_ok=True)
        return OUTPUT_DIR
    except Exception:
        # Fallback for locked-down environments
        fallback = Path("/tmp/generated_pdfs")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# Footer + page-break helpers (applies footer on EVERY page)
def _draw_footer(c, width, margin):
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, margin - 20, FOOTER_TEXT)
    # Optional page number on the right:
    page_num = c.getPageNumber()
    text = f"Page {page_num}"
    tw = c.stringWidth(text, "Helvetica-Oblique", 9)
    c.drawString(width - margin - tw, margin - 20, text)


def _page_break(c, width, height, margin):
    # draw footer on current page, then advance
    _draw_footer(c, width, margin)
    c.showPage()


# Text / layout helpers
def _wrap_text(canvas_obj, text, max_width, font_name="Helvetica", font_size=11):
    """Split text into lines that fit within max_width at the given font."""
    words = text.split(" ")
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        if canvas_obj.stringWidth(test, font_name, font_size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            if canvas_obj.stringWidth(w, font_name, font_size) <= max_width:
                current = w
            else:
                # Hard-break very long word
                piece = ""
                for ch in w:
                    if (
                        canvas_obj.stringWidth(piece + ch, font_name, font_size)
                        <= max_width
                    ):
                        piece += ch
                    else:
                        if piece:
                            lines.append(piece)
                        piece = ch
                current = piece
    if current:
        lines.append(current)
    return lines


def _draw_paragraph(
    c,
    text,
    x,
    y,
    max_width,
    width,
    height,
    leading=PARAGRAPH_LEADING,
    font="Helvetica",
    size=11,
    bottom_margin=MARGIN,
    spacing=SECTION_SPACING,
):
    """Draw a wrapped paragraph and return updated y with section spacing."""
    c.setFont(font, size)
    lines = []
    for block in text.split("\n"):
        lines.extend(_wrap_text(c, block, max_width, font, size))
    for line in lines:
        if y - leading < bottom_margin:
            _page_break(c, width, height, bottom_margin)
            y = height - bottom_margin
            c.setFont(font, size)
        c.drawString(x, y, line)
        y -= leading
    return y - spacing


def _draw_section_title(
    c, title, x, y, width, height, bottom_margin=MARGIN, spacing=SECTION_SPACING
):
    if y - 22 < bottom_margin:
        _page_break(c, width, height, bottom_margin)
        y = height - bottom_margin
    c.setFont("Helvetica-Bold", 12)

    if "Feature contributions" in title:
        text_width = c.stringWidth(title, "Helvetica-Bold", 12)
        c.setFillColorRGB(1, 1, 0)  # yellow
        c.rect(x - 2, y - 2, text_width + 4, 16, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)  # reset

    c.drawString(x, y, title)
    return y - spacing


def _ensure_space_or_newpage(c, y, needed, width, height, bottom_margin=MARGIN):
    """Ensure there is vertical space 'needed', else insert a page break."""
    if y - needed < bottom_margin:
        _page_break(c, width, height, bottom_margin)
        return height - bottom_margin
    return y


def _draw_key_value_rows(
    c,
    pairs,
    x,
    y,
    col_gap,
    max_width_right,
    width,
    height,
    leading=PARAGRAPH_LEADING,
    bottom_margin=MARGIN,
    spacing=SECTION_SPACING,
):
    """Draw 'Key: Value' rows with wrapping on the value side; return updated y with spacing."""
    left_font = ("Helvetica-Bold", 10)
    right_font = ("Helvetica", 10)
    for left, right in pairs:
        y = _ensure_space_or_newpage(c, y, leading, width, height, bottom_margin)
        c.setFont(*left_font)
        c.drawString(x, y, left)
        c.setFont(*right_font)
        right_lines = _wrap_text(
            c, right, max_width_right, right_font[0], right_font[1]
        )

        for rl in right_lines:
            y = _ensure_space_or_newpage(c, y, leading, width, height, bottom_margin)
            c.setFont(*right_font)
            c.drawString(x + col_gap, y, rl)
            y -= leading
    return y - spacing


# Main
def generate_pdf(prediction_label, prob_percentage, explanation_dict):
    """
    prediction_label: 'Default' or 'Non-Default' (string)
    prob_percentage: e.g., '85.62%' (string with %)
    explanation_dict: dict of {feature_name: shap_value}, may include 20+ keys
    """
    # # Dev mode
    # filename = "credit_report.pdf"
    # filepath = os.path.join(OUTPUT_DIR, filename)

    # Docker mode
    out_dir = _ensure_output_dir()
    filename = "credit_report.pdf"
    filepath = out_dir / filename

    c = canvas.Canvas(str(filepath), pagesize=letter)
    c.setTitle("Credit Risk Assessment Report")
    c.setAuthor("FlexiLoan Company")
    c.setSubject("Credit Risk Prediction Report")
    c.setKeywords("credit, risk, assessment, report")

    width, height = letter
    x = MARGIN
    y = height - MARGIN
    max_text_width = width - 2 * MARGIN

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x, y, "Credit Risk Assessment Report")
    y -= 45

    c.setFont("Helvetica", 11)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(x, y, f"Generated on: {now_str} (GMT+7)")
    y -= 25

    result_text = f"{prediction_label}{prob_percentage}"
    label_text = "Prediction Result: "

    # Draw the prefix normally
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, label_text)

    # Compute highlight position
    offset = c.stringWidth(label_text, "Helvetica-Bold", 12)
    highlight_x = x + offset + 5
    highlight_y = y

    c.setFont("Helvetica-Bold", 12)
    text_width = c.stringWidth(result_text, "Helvetica-Bold", 12)

    c.setFillColorRGB(1, 1, 0)  # yellow
    c.rect(highlight_x - 2, highlight_y - 2, text_width + 4, 16, fill=1, stroke=0)

    c.setFillColorRGB(0, 0, 0)  # reset
    c.drawString(highlight_x, highlight_y, result_text)
    y -= 45

    # Explanations
    for title, para in EXPLANATION_PARAGRAPHS:
        y = _draw_section_title(c, title, x, y, width, height)
        y = _draw_paragraph(c, para, x, y, max_text_width, width, height)

    # Force new page before Feature contributions
    _page_break(c, width, height, MARGIN)
    y = height - MARGIN
    x = MARGIN

    # Feature contributions (SHAP)
    y = _draw_section_title(
        c, "Feature contributions for this applicant (SHAP)", x, y, width, height
    )
    legend = "Positive value → higher risk (pushes toward Default) \nNegative value → lower risk (pushes toward Non-Default)"
    y = _draw_paragraph(
        c,
        legend,
        x,
        y,
        max_text_width,
        width,
        height,
        font="Helvetica-Oblique",
        size=11,
    )

    # Sort by absolute SHAP value (desc) and take top 20
    explanation_dict = dict(
        sorted(explanation_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    top_items = list(explanation_dict.items())[:20]

    display_pairs = []
    for feat, val in top_items:
        direction = "↑ risk" if val > 0 else ("↓ risk" if val < 0 else "no effect")
        right = f"{val:+.4f} ({direction})"
        display_pairs.append((f"{feat}:", right))

    col_gap = 180
    max_width_right = max_text_width - col_gap
    y = _draw_key_value_rows(
        c,
        display_pairs,
        x,
        y,
        col_gap,
        max_width_right,
        width,
        height,
        leading=PARAGRAPH_LEADING,
        bottom_margin=MARGIN,
        spacing=SECTION_SPACING,
    )

    # Inserted explanatory note about SHAP
    shap_note = (
        "Note: SHAP values show how a feature’s actual value pushes risk up or down relative to others.\n"
        "• If EXT_SOURCE_1 = 0.7 and SHAP = +0.3 (↑ risk), then despite 0.7 being a decent score, for this applicant it’s low compared to others, so it increases risk.\n"
        "• If EXT_SOURCE_1 = 0.7 and SHAP = –0.3 (↓ risk), then 0.7 is relatively high compared to others, so it reduces risk.\n"
    )
    y = _draw_paragraph(
        c, shap_note, x, y, max_text_width, width, height, font="Helvetica", size=11
    )

    # Force new page before Feature glossary
    _page_break(c, width, height, MARGIN)
    y = height - MARGIN
    x = MARGIN

    y = _draw_section_title(
        c, "Feature name glossary (plain-English)", x, y, width, height
    )
    intro = "Short descriptions of each field to help interpretation."
    y = _draw_paragraph(c, intro, x, y, max_text_width, width, height, size=11)

    glossary_pairs = []
    for key_label, _ in display_pairs:
        feat = key_label.rstrip(":")
        desc = FEATURE_GLOSSARY.get(feat, "Description not available.")
        glossary_pairs.append((f"{feat}:", desc))

    y = _draw_key_value_rows(
        c,
        glossary_pairs,
        x,
        y,
        col_gap,
        max_width_right,
        width,
        height,
        leading=PARAGRAPH_LEADING,
        bottom_margin=MARGIN,
        spacing=SECTION_SPACING,
    )

    # Footer on LAST page
    _draw_footer(c, width, MARGIN)
    c.save()
    return filename
