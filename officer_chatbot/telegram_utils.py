from pathlib import Path # top
import os
import io
import json
import math
import tiktoken
import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


# Load env
load_dotenv(dotenv_path=ENV_PATH, override=True)


# External tokens/keys
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Model & limits (override-able via env)
MODEL_NAME = os.getenv("OPENAI_MODEL")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))


# Pricing (USD per 1K tokens)
PRICE_PER_1K_INPUT_TOKENS = float(os.getenv("PRICE_PER_1K_INPUT_TOKENS", "0.0005"))
PRICE_PER_1K_OUTPUT_TOKENS = float(os.getenv("PRICE_PER_1K_OUTPUT_TOKENS", "0.0015"))


# RAG/vectorstore & keywords
VECTORSTORE_PATH = PROJECT_ROOT / "rag_vectorstore"
KEYWORDS_PATH = PROJECT_ROOT / "info" / "keywords.json"


# Token utilities
def count_tokens(text: str, model: str = MODEL_NAME) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text or ""))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000) * PRICE_PER_1K_INPUT_TOKENS
        + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT_TOKENS,
        6,
    )


# Loan keywords 
with open(KEYWORDS_PATH, "r") as _f:
    _keywords = json.load(_f)
LOAN_KEYWORDS = [kw.lower() for kw in _keywords.get("loan_keywords", [])]


def contains_loan_keyword(message: str) -> bool:
    msg = (message or "").lower()
    return any(word in msg for word in LOAN_KEYWORDS)


# ====================================================


def generate_pie_chart_target(df):
    try:
        # Mapping 0/1 to readable labels
        label_map = {
            0: "Non-Default",
            1: "Default"
        }

        # Clean input
        labels_raw = df.iloc[:, 0].fillna("Unknown").astype(int)
        sizes = df.iloc[:, 1].astype(int)
        labels = [label_map.get(val, str(val)) for val in labels_raw]

        total = sum(sizes)

        # Custom label
        def make_autopct(values):
            def my_autopct(pct):
                absolute = int(round(pct / 100.0 * sum(values)))
                return f'{pct:.1f}% ({absolute:,})'
            return my_autopct

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=make_autopct(sizes),
            startangle=90,
            colors=["#72B5D7", "#E88A89"],
            textprops={'fontsize': 12}
        )

        plt.setp(autotexts, size=12, weight="bold")
        ax.axis('equal')
        plt.title("Defaults vs Non-Defaults", fontsize=15, weight="bold", pad=30)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf
    except Exception as e:
        print(f"[Chart Error] {e}")
        return None

def generate_bar(df: pd.DataFrame) -> BytesIO | None:
    if df.empty or df.shape[1] != 2:
        return None

    x_col, y_col = df.columns

    # Ensure y is numeric
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df.dropna(subset=[y_col], inplace=True)
    if df.empty:
        return None

    # Sort by y ascending
    df = df.sort_values(by=y_col, ascending=True)

    # Create horizontal bar chart
    plt.clf()
    fig, ax = plt.subplots()
    bars = ax.barh(df[x_col], df[y_col], color="blue")

    # Add value labels to the right of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=9
        )

    # Style
    ax.set_title(f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf

def generate_histogram(df: pd.DataFrame) -> BytesIO | None:
    # expects a single numeric column
    if df.empty or df.shape[1] != 1:
        return None

    col = df.columns[0]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=[col], inplace=True)
    if df.empty:
        return None

    plt.clf()
    fig, ax = plt.subplots()
    
    ## ===== Original Data – No Capping Applied
    # ax.hist(df[col], bins=20, edgecolor="white", color="blue")
    ## =====

    ## ===== 99.5% Data (Capping extreme outliers at the top 0.5%)
    lo, hi = np.nanpercentile(df[col], [1, 99])  # or [0, 99.5] 
    ax.hist(df[col].clip(lo, hi), bins=20, edgecolor="white", color="blue", range=(lo, hi))
    ax.set_xlim(lo, hi)
    ax.margins(x=0)
    ## =====

    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='y', left=False, labelleft=False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf

def generate_scatter(df: pd.DataFrame) -> BytesIO | None:
    # expects two numeric columns
    if df.empty or df.shape[1] != 2:
        return None

    x_col, y_col = df.columns
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df.dropna(subset=[x_col, y_col], inplace=True)
    if df.empty:
        return None

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], s=20, color="blue")

    # add a simple linear "guide" line (least squares)
    try:
        coeffs = np.polyfit(df[x_col], df[y_col], 1)
        poly = np.poly1d(coeffs)
        xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(xs, poly(xs), linewidth=1)
    except Exception:
        pass  # if fit fails, just show points

    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf

def generate_pdf(df, max_cols_per_table=5, meta=None, loan_officer_name="Loan Officer"):
    """
    Generates a PDF with default metadata tailored for loan officers.
    Pass a `meta` dict to override any defaults.
    """
    # --- default, loan-officer-friendly metadata ---
    default_meta = {
        "title":   "Loan Data Summary",
        "author":  loan_officer_name,
        "subject": "Summary",
        "keywords": ["loan", "underwriting", "application", "risk", "credit"],
        "creator": "Loan Ops PDF Generator"
    }
    if meta:
        default_meta.update(meta)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    num_cols = len(df.columns)
    num_tables = math.ceil(num_cols / max_cols_per_table)

    for i in range(num_tables):
        start = i * max_cols_per_table
        end = start + max_cols_per_table
        subset = df.iloc[:, start:end]

        data = [subset.columns.tolist()] + subset.values.tolist()

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

    # Set PDF metadata on creation
    def _set_pdf_metadata(canvas, _doc):
        title = default_meta.get("title")
        if title: canvas.setTitle(title)
        author = default_meta.get("author")
        if author: canvas.setAuthor(author)
        subject = default_meta.get("subject")
        if subject: canvas.setSubject(subject)
        keywords = default_meta.get("keywords")
        if keywords:
            if isinstance(keywords, (list, tuple)):
                keywords = ", ".join(keywords)
            canvas.setKeywords(keywords)
        creator = default_meta.get("creator")
        if creator: canvas.setCreator(creator)

    doc.build(elements, onFirstPage=_set_pdf_metadata, onLaterPages=_set_pdf_metadata)
    buffer.seek(0)
    return buffer

