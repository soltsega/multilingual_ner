import html
from typing import Any

import streamlit as st
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multilingual KYC NER",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# APPLICATION CONSTANTS
# ============================================================

MODEL_ID = "SoloCode/multilingual-ner"

APP_NAME = "Multilingual KYC NER"

APP_SUBTITLE = (
    "A multilingual named entity recognition system for extracting "
    "identity-related information from KYC text."
)

SUPPORTED_LANGUAGES = [
    "Add your supported languages here"
]

ENTITY_TYPES = [
    "Add your entity types here"
]

DATASET_INFORMATION = {
    "Primary dataset": "Replace with your actual dataset name",
    "Additional dataset": "Replace with your actual dataset name",
    "Task": "Named Entity Recognition",
    "Domain": "KYC / identity-related text",
}

# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    ("NER Analysis", "✦"),
    ("Methodology", "◈"),
    ("Dataset", "▦"),
    ("Model", "◎"),
    ("System Architecture", "⌘"),
    ("Limitations", "△"),
    ("Future Improvements", "↗"),
]

if "page" not in st.session_state:
    st.session_state.page = "NER Analysis"

page = st.session_state.page

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 90% 0%,
                rgba(99, 102, 241, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 0% 20%,
                rgba(14, 165, 233, 0.06),
                transparent 25%
            ),
            #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ========================================================
       TOP BRAND BAR
       ======================================================== */

    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.25rem;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(12px);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: white;
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        box-shadow: 0 7px 18px rgba(79, 70, 229, 0.25);
    }

    .brand-name {
        font-weight: 800;
        font-size: 1rem;
        color: #0f172a;
        line-height: 1.2;
    }

    .brand-caption {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.5rem 2.5rem;
        margin-bottom: 1.25rem;
        border-radius: 24px;
        color: white;
        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(255,255,255,0.18),
                transparent 25%
            ),
            linear-gradient(135deg, #312e81 0%, #4f46e5 48%, #7c3aed 100%);
        box-shadow: 0 18px 45px rgba(79, 70, 229, 0.20);
    }

    .hero-eyebrow {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.8;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.75rem;
        font-weight: 850;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 820px;
        color: rgba(255,255,255,0.86);
    }

    .hero-badge {
        display: inline-block;
        margin-top: 1.25rem;
        padding: 0.42rem 0.72rem;
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        font-size: 0.78rem;
        font-weight: 650;
    }

    /* ========================================================
       NAVIGATION
       ======================================================== */

    .nav-heading {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 800;
        color: #64748b;
        margin: 0.7rem 0 0.55rem 0.15rem;
    }

    /* Streamlit button container */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 12px !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        border-color: #818cf8 !important;
        color: #4338ca !important;
        background: #eef2ff !important;
        transform: translateY(-1px);
    }

    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        ) !important;
        border: 1px solid #4f46e5 !important;
        color: white !important;
        font-weight: 800 !important;
        min-height: 44px !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.20);
    }

    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0f172a;
        margin-top: 1.3rem;
        margin-bottom: 0.45rem;
    }

    .section-description {
        color: #64748b;
        line-height: 1.75;
        max-width: 900px;
        margin-bottom: 1.3rem;
    }

    .page-kicker {
        color: #4f46e5;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 850;
        margin-bottom: 0.3rem;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .info-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.045);
    }

    .info-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.02rem;
        color: #0f172a;
    }

    .info-card p {
        color: #64748b;
        line-height: 1.65;
        margin-bottom: 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 125px;
        box-shadow: 0 5px 20px rgba(15,23,42,0.04);
    }

    .metric-icon {
        font-size: 1.35rem;
        margin-bottom: 0.65rem;
    }

    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 750;
    }

    .metric-value {
        font-size: 1.18rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
    }

    .metric-description {
        color: #64748b;
        font-size: 0.83rem;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    /* ========================================================
       ANALYSIS
       ======================================================== */

    .analysis-shell {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 8px 30px rgba(15,23,42,0.045);
    }

    .input-label {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.45rem;
    }

    .example-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 0.5rem;
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin: 1.5rem 0 0.8rem 0;
    }

    .result-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
    }

    .result-count {
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.75rem;
        font-weight: 800;
    }

    /* ========================================================
       ENTITY HIGHLIGHTING
       ======================================================== */

    .entity-text {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem;
        line-height: 2.25;
        font-size: 1rem;
        color: #1e293b;
        box-shadow: 0 5px 20px rgba(15,23,42,0.035);
    }

    .entity {
        display: inline-block;
        padding: 0.13rem 0.45rem;
        margin: 0 0.08rem;
        border-radius: 7px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #312e81;
        font-weight: 650;
    }

    .entity-label {
        font-size: 0.65rem;
        font-weight: 850;
        color: #4f46e5;
        margin-left: 0.32rem;
        vertical-align: middle;
    }

    /* ========================================================
       NOTICE
       ======================================================== */

    .notice {
        border-left: 4px solid #4f46e5;
        background: #eef2ff;
        padding: 1rem 1.1rem;
        border-radius: 0 12px 12px 0;
        color: #3730a3;
        line-height: 1.6;
    }

    .warning {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
        padding: 1rem 1.1rem;
        border-radius: 0 12px 12px 0;
        color: #92400e;
        line-height: 1.6;
    }

    .success-box {
        border-left: 4px solid #10b981;
        background: #ecfdf5;
        padding: 1rem 1.1rem;
        border-radius: 0 12px 12px 0;
        color: #065f46;
        line-height: 1.6;
    }

    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline-step {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        min-height: 135px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.035);
    }

    .pipeline-number {
        font-size: 0.7rem;
        color: #6366f1;
        font-weight: 850;
        letter-spacing: 0.08em;
    }

    .pipeline-title {
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.35rem;
    }

    .pipeline-description {
        color: #64748b;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    /* ========================================================
       ARCHITECTURE
       ======================================================== */

    .architecture-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem 1.2rem;
        box-shadow: 0 5px 18px rgba(15,23,42,0.035);
    }

    .architecture-number {
        min-width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: #eef2ff;
        color: #4338ca;
        font-weight: 850;
        font-size: 0.8rem;
    }

    .architecture-title {
        font-weight: 800;
        color: #0f172a;
    }

    .architecture-description {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.2rem;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-card {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #64748b;
        font-size: 0.8rem;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.6rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
        }

        .brand-bar {
            padding: 0.75rem;
        }

        .status-pill {
            display: none;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ner_pipeline():
    """
    Load the Hugging Face tokenizer and model once and cache them.

    The model is intentionally loaded only when inference is requested.
    Streamlit reruns the script when widgets are interacted with, but
    cache_resource prevents reconstruction of the model during normal
    reruns.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID
    )

    device = 0 if torch.cuda.is_available() else -1

    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    return ner_pipeline


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any) -> float:
    """Convert a confidence value safely to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_entity_label(label: str) -> str:
    """
    Make model labels easier to read.

    Example:
        B-PERSON -> PERSON
        PER -> PER
    """

    if not label:
        return "ENTITY"

    label = label.replace("B-", "")
    label = label.replace("I-", "")

    return label


def highlight_entities(text: str, entities: list[dict]) -> str:
    """
    Highlight detected entities in the original text.

    Uses character offsets returned by the Hugging Face pipeline.
    """

    if not entities:
        return html.escape(text)

    sorted_entities = sorted(
        entities,
        key=lambda item: item.get("start", 0)
    )

    output = []
    cursor = 0

    for entity in sorted_entities:

        start = entity.get("start")
        end = entity.get("end")

        if start is None or end is None:
            continue

        if start < cursor:
            continue

        output.append(
            html.escape(text[cursor:start])
        )

        value = html.escape(
            text[start:end]
        )

        label = html.escape(
            format_entity_label(
                entity.get("entity_group", "ENTITY")
            )
        )

        output.append(
            f"""
            <span class="entity">
                {value}
                <span class="entity-label">{label}</span>
            </span>
            """
        )

        cursor = end

    output.append(
        html.escape(text[cursor:])
    )

    return "".join(output)


def confidence_level(score: float) -> str:

    if score >= 0.90:
        return "High"

    if score >= 0.70:
        return "Moderate"

    return "Lower"


# ============================================================
# TOP BRAND BAR
# ============================================================

st.markdown(
    """
    <div class="brand-bar">
        <div class="brand-left">
            <div class="brand-icon">◈</div>
            <div>
                <div class="brand-name">
                    Multilingual KYC NER
                </div>
                <div class="brand-caption">
                    INSA Summer Camp · Research & Demonstration
                </div>
            </div>
        </div>

        <div class="status-pill">
            <span class="status-dot"></span>
            Model Available
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-eyebrow">
            Multilingual Information Extraction
        </div>

        <div class="hero-title">
            {APP_NAME}
        </div>

        <div class="hero-subtitle">
            {APP_SUBTITLE}
        </div>

        <div class="hero-badge">
            Transformer-based Named Entity Recognition
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NAVIGATION
# ============================================================

st.markdown(
    '<div class="nav-heading">Explore the system</div>',
    unsafe_allow_html=True,
)

# 4 + 3 navigation layout
nav_rows = [
    PAGES[:4],
    PAGES[4:],
]

for row in nav_rows:

    columns = st.columns(len(row))

    for column, (label, icon) in zip(columns, row):

        with column:

            is_active = page == label

            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):

                st.session_state.page = label
                st.rerun()

st.divider()

# ============================================================
# PAGE HEADER
# ============================================================

page_descriptions = {
    "NER Analysis": (
        "Run the trained model on identity-related text and inspect "
        "the entities, confidence scores, and annotated output."
    ),
    "Methodology": (
        "Understand the data preparation, label alignment, merging, "
        "tokenization, fine-tuning, and evaluation workflow."
    ),
    "Dataset": (
        "Review the role of the training data, dataset merging, "
        "label compatibility, and important data constraints."
    ),
    "Model": (
        "Understand the trained multilingual transformer model and "
        "its role as the NER inference component."
    ),
    "System Architecture": (
        "Explore how the interface, inference pipeline, tokenizer, "
        "model, and output layer work together."
    ),
    "Limitations": (
        "Understand the boundaries of the current system and how "
        "its predictions should be interpreted."
    ),
    "Future Improvements": (
        "Review practical improvements across data, modeling, "
        "document processing, evaluation, and deployment."
    ),
}

st.markdown(
    f'<div class="page-kicker">{page}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="section-description">'
    f'{page_descriptions[page]}'
    f'</div>',
    unsafe_allow_html=True,
)

# ============================================================
# NER ANALYSIS
# ============================================================

if page == "NER Analysis":

    st.markdown(
        """
        <div class="analysis-shell">
            <div class="section-title" style="margin-top:0;">
                Analyze KYC Text
            </div>

            <div class="section-description">
                Enter identity-related text below. The trained model
                will identify candidate entity spans and provide a
                confidence score for each prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_text = (
        "Full Name: Abebe Kebede\n"
        "Date of Birth: 12/04/1998\n"
        "Passport Number: ET1234567"
    )

    text = st.text_area(
        "Input text",
        value="",
        placeholder=example_text,
        height=220,
        help="Enter the text you want the NER model to analyze.",
        label_visibility="visible",
    )

    st.markdown(
        f"""
        <div class="example-box">
            <strong>Example format</strong><br>
            Full Name: Abebe Kebede<br>
            Date of Birth: 12/04/1998<br>
            Passport Number: ET1234567
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    analyze_col, clear_col, spacer = st.columns(
        [2.0, 1.0, 4.0]
    )

    with analyze_col:

        analyze = st.button(
            "✦  Analyze Text",
            type="primary",
            use_container_width=True,
        )

    with clear_col:

        clear = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear:

        st.rerun()

    if analyze:

        if not text.strip():

            st.warning(
                "Please enter text before starting the analysis."
            )

        else:

            try:

                with st.spinner(
                    "Loading the multilingual model and running inference..."
                ):

                    ner = load_ner_pipeline()

                    results = ner(text)

                st.markdown(
                    f"""
                    <div class="success-box">
                        <strong>Analysis completed.</strong>
                        {len(results)} entity span(s) detected.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if results:

                    st.markdown(
                        """
                        <div class="result-header">
                            <div class="result-title">
                                Detected Entities
                            </div>

                            <div class="result-count">
                                Entity spans detected
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    rows = []

                    for item in results:

                        score = safe_float(
                            item.get("score", 0)
                        )

                        rows.append(
                            {
                                "Entity": format_entity_label(
                                    item.get(
                                        "entity_group",
                                        "ENTITY"
                                    )
                                ),
                                "Text": item.get(
                                    "word",
                                    ""
                                ),
                                "Confidence": f"{score:.2%}",
                                "Interpretation": confidence_level(
                                    score
                                ),
                            }
                        )

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown(
                        '<div class="section-title">'
                        'Annotated Text'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    highlighted = highlight_entities(
                        text,
                        results,
                    )

                    st.markdown(
                        f'<div class="entity-text">'
                        f'{highlighted}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Confidence indicates the model's prediction "
                        "score for the detected span. It should not "
                        "be interpreted as a probability that the "
                        "identity information is factually correct."
                    )

                else:

                    st.info(
                        "No entities were detected in the provided text."
                    )

            except Exception as exc:

                st.error(
                    "The model could not process this input."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(exc)
                    )

    st.write("")

    st.markdown(
        """
        <div class="notice">
            <strong>Important:</strong>
            NER identifies candidate entities in text. It does not
            independently verify the authenticity of an identity,
            document, or piece of information.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="section-title">End-to-End Methodology</div>',
        unsafe_allow_html=True,
    )

    methodology = [
        (
            "1",
            "Dataset Selection",
            "Relevant annotated NER datasets were selected according "
            "to entity coverage, language coverage, annotation quality, "
            "and relevance to the target task.",
        ),
        (
            "2",
            "Dataset Analysis",
            "The datasets were examined for label definitions, annotation "
            "conventions, language distribution, and differences in data "
            "characteristics.",
        ),
        (
            "3",
            "Label Alignment",
            "Labels from different sources must represent compatible "
            "entity meanings before examples are combined. This reduces "
            "the risk of contradictory supervision.",
        ),
        (
            "4",
            "Dataset Merging",
            "Compatible training examples can be combined to increase "
            "the diversity and coverage of the training corpus.",
        ),
        (
            "5",
            "Data Splitting",
            "The resulting corpus is divided into training, validation, "
            "and test partitions so model development and final "
            "evaluation remain conceptually separate.",
        ),
        (
            "6",
            "Tokenization",
            "Text is converted into the representation required by the "
            "transformer while preserving the relationship between "
            "tokens and entity labels.",
        ),
        (
            "7",
            "Fine-tuning",
            "A pretrained multilingual transformer is adapted to the "
            "NER task using the prepared labeled corpus.",
        ),
        (
            "8",
            "Evaluation",
            "Model predictions should be evaluated using entity-level "
            "precision, recall, and F1, preferably broken down by "
            "language and entity type.",
        ),
    ]

    cols = st.columns(2)

    for index, (number, title, description) in enumerate(
        methodology
    ):

        with cols[index % 2]:

            st.markdown(
                f"""
                <div class="info-card"
                     style="margin-bottom:1rem;">

                    <div class="architecture-number">
                        {number}
                    </div>

                    <h3 style="margin-top:0.8rem;">
                        {title}
                    </h3>

                    <p>
                        {description}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Why Dataset Merging Matters</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">

                <h3>Potential Benefit</h3>

                <p>
                    A merged corpus can provide broader linguistic
                    and entity coverage than an individual dataset.
                    This can expose the model to more varied patterns
                    during training.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

                <h3>What Must Be Controlled</h3>

                <p>
                    Merging is not automatically beneficial. Differences
                    in label definitions, annotation quality, language
                    distribution, and domain can introduce noise or
                    inconsistent supervision.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">BIO Tagging</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">

        <p>
        The BIO scheme represents entity boundaries at the token level.
        <strong>B</strong> identifies the beginning of an entity,
        <strong>I</strong> identifies a continuation of an entity, and
        <strong>O</strong> represents a token outside an entity.
        </p>

        <br>

        <code>
        Solomon → B-PERSON<br>
        Tsega → I-PERSON<br>
        lives → O<br>
        in → O<br>
        Addis → B-LOCATION<br>
        Ababa → I-LOCATION
        </code>

        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">Training Data</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">

                <h3>Dataset Sources</h3>

                <p>
                    The exact dataset names, versions, licenses,
                    language coverage, and sample counts should be
                    documented here according to the final training
                    notebook.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

                <h3>Why Multiple Sources?</h3>

                <p>
                    Combining compatible datasets can increase the
                    diversity of examples available during training.
                    The benefit depends on consistent entity definitions
                    and annotation quality.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Dataset Configuration</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        [
            {
                "Property": key,
                "Value": value,
            }
            for key, value in DATASET_INFORMATION.items()
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Dataset Quality Checks</div>',
        unsafe_allow_html=True,
    )

    checks = [
        "Equivalent labels should have equivalent meanings.",
        "Annotation boundaries should be compatible.",
        "Entity types should be mapped consistently.",
        "One dataset should not unintentionally dominate the merged distribution.",
        "Language-specific differences should be preserved.",
        "Duplicate and near-duplicate examples should be investigated.",
        "Licensing conditions should be compatible.",
    ]

    for check in checks:

        st.markdown(
            f"""
            <div class="info-card"
                 style="margin-bottom:0.6rem;padding:0.9rem 1rem;">

                <strong>✓</strong>
                &nbsp; {check}

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="warning" style="margin-top:1rem;">
            A merged dataset is useful only when the resulting
            supervision remains coherent. Increasing the number
            of examples alone does not guarantee better model
            performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL
# ============================================================

elif page == "Model":

    st.markdown(
        '<div class="section-title">Trained NER Model</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-icon">◎</div>

                <div class="metric-label">
                    Model Type
                </div>

                <div class="metric-value">
                    Transformer
                </div>

                <div class="metric-description">
                    Fine-tuned for token classification.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-icon">◈</div>

                <div class="metric-label">
                    Task
                </div>

                <div class="metric-value">
                    NER
                </div>

                <div class="metric-description">
                    Multilingual named entity recognition.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-icon">↗</div>

                <div class="metric-label">
                    Model Hub
                </div>

                <div class="metric-value">
                    Hugging Face
                </div>

                <div class="metric-description">
                    Published model artifact.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Model Identifier</div>',
        unsafe_allow_html=True,
    )

    st.code(
        MODEL_ID,
        language="text",
    )

    st.markdown(
        """
        <div class="notice">
            The application loads the trained model from the Hugging
            Face Model Hub only when inference is requested. The
            resource is cached using Streamlit's resource cache so
            normal widget interactions do not reconstruct the model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Inference Flow</div>',
        unsafe_allow_html=True,
    )

    model_steps = [
        (
            "01",
            "Input Text",
            "Identity-related text is supplied by the user.",
        ),
        (
            "02",
            "Tokenizer",
            "The text is converted into model-compatible tokens.",
        ),
        (
            "03",
            "Transformer",
            "The fine-tuned multilingual model generates token-level predictions.",
        ),
        (
            "04",
            "Aggregation",
            "Token predictions are combined into entity spans.",
        ),
        (
            "05",
            "Output",
            "Entities and confidence scores are presented to the user.",
        ),
    ]

    for number, title, description in model_steps:

        st.markdown(
            f"""
            <div class="architecture-card"
                 style="margin-bottom:0.7rem;">

                <div class="architecture-number">
                    {number}
                </div>

                <div>
                    <div class="architecture-title">
                        {title}
                    </div>

                    <div class="architecture-description">
                        {description}
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

elif page == "System Architecture":

    st.markdown(
        '<div class="section-title">System Architecture</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        The application separates presentation, model loading,
        tokenization, inference, and output formatting. This allows
        the interface and supporting documentation to evolve without
        changing the trained model itself.
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture = [
        (
            "User Interface",
            "Streamlit provides the interaction layer for text input, "
            "results, visualization, and project documentation.",
        ),
        (
            "Inference Layer",
            "The application sends input text to the NER pipeline and "
            "receives entity spans and confidence scores.",
        ),
        (
            "Tokenizer",
            "The tokenizer converts text into the representation expected "
            "by the transformer model.",
        ),
        (
            "Multilingual NER Model",
            f"{MODEL_ID} provides the fine-tuned token classification model.",
        ),
        (
            "Output Layer",
            "Predictions are transformed into human-readable entity "
            "tables and annotated text.",
        ),
    ]

    for index, (title, description) in enumerate(
        architecture,
        1,
    ):

        st.markdown(
            f"""
            <div class="architecture-card"
                 style="margin-bottom:0.75rem;">

                <div class="architecture-number">
                    {index:02d}
                </div>

                <div>

                    <div class="architecture-title">
                        {title}
                    </div>

                    <div class="architecture-description">
                        {description}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Deployment Flow</div>',
        unsafe_allow_html=True,
    )

    deployment = [
        ("01", "GitHub", "Application source code"),
        ("02", "Streamlit", "Hosted user interface"),
        ("03", "Hugging Face", "Trained model artifact"),
        ("04", "Inference", "Entity extraction"),
    ]

    cols = st.columns(4)

    for column, (number, title, description) in zip(
        cols,
        deployment,
    ):

        with column:

            st.markdown(
                f"""
                <div class="pipeline-step">

                    <div class="pipeline-number">
                        {number}
                    </div>

                    <div class="pipeline-title">
                        {title}
                    </div>

                    <div class="pipeline-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.markdown(
        '<div class="section-title">Constraints & Limitations</div>',
        unsafe_allow_html=True,
    )

    limitations = [
        (
            "Dataset coverage",
            "Performance depends on how well the training data represents "
            "the languages, entity types, writing styles, and document "
            "conditions encountered in deployment.",
        ),
        (
            "Annotation consistency",
            "Combining datasets with different annotation conventions can "
            "introduce inconsistent supervision even when labels have "
            "similar names.",
        ),
        (
            "Domain shift",
            "Performance may decrease when deployment text differs "
            "substantially from the training data.",
        ),
        (
            "Document quality",
            "If text is obtained through OCR, recognition errors may be "
            "propagated into the NER stage.",
        ),
        (
            "Confidence interpretation",
            "Model confidence is not equivalent to factual verification. "
            "A highly confident prediction can still be incorrect.",
        ),
        (
            "KYC verification",
            "Entity extraction alone does not establish that a person's "
            "identity or document is genuine.",
        ),
        (
            "Resource requirements",
            "The transformer model is relatively large and therefore "
            "requires substantially more memory than a lightweight "
            "rule-based extractor.",
        ),
    ]

    for title, description in limitations:

        with st.expander(
            f"△  {title}",
            expanded=False,
        ):

            st.write(description)

    st.markdown(
        """
        <div class="warning" style="margin-top:1.2rem;">
            <strong>Core constraint:</strong>
            The system is an information-extraction component.
            It should not be treated as a complete KYC verification,
            authentication, or fraud-detection system.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FUTURE IMPROVEMENTS
# ============================================================

elif page == "Future Improvements":

    st.markdown(
        '<div class="section-title">Potential Improvements</div>',
        unsafe_allow_html=True,
    )

    improvements = {
        "Data": [
            "Expand multilingual KYC-specific training data.",
            "Improve representation of underrepresented languages.",
            "Increase entity-type coverage.",
            "Audit annotation consistency.",
            "Perform systematic duplicate detection.",
        ],
        "Model": [
            "Compare alternative multilingual transformer architectures.",
            "Investigate domain-adaptive pretraining.",
            "Evaluate parameter-efficient fine-tuning.",
            "Investigate model quantization for deployment.",
        ],
        "Document Processing": [
            "Integrate OCR for scanned documents.",
            "Use layout-aware document models.",
            "Preserve relationships between fields and values.",
            "Handle tables and structured document regions.",
        ],
        "Evaluation": [
            "Report entity-level precision, recall, and F1.",
            "Evaluate each language independently.",
            "Evaluate each entity type independently.",
            "Perform cross-domain evaluation.",
            "Create a structured error taxonomy.",
        ],
        "Deployment": [
            "Optimize inference latency.",
            "Investigate CPU/GPU deployment trade-offs.",
            "Cache model resources.",
            "Consider model serving separate from the UI.",
        ],
    }

    categories = list(improvements.items())

    cols = st.columns(2)

    for index, (category, items) in enumerate(categories):

        with cols[index % 2]:

            items_html = "".join(
                f"<li>{item}</li>"
                for item in items
            )

            st.markdown(
                f"""
                <div class="info-card"
                     style="margin-bottom:1rem;">

                    <h3>{category}</h3>

                    <ul style="
                        color:#64748b;
                        line-height:1.8;
                        padding-left:1.2rem;
                    ">
                        {items_html}
                    </ul>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="notice">
            Future development should target measurable weaknesses
            rather than adding functionality solely for presentation.
            The strongest improvements should be validated through
            better data, stronger evaluation, and improved deployment
            reliability.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-card">

        <strong>Multilingual KYC NER</strong><br>

        Research and demonstration interface developed as part of
        the Ethiopian Information Network Security Agency (INSA)
        Summer Camp.

        <br><br>

        Model:
        <strong>SoloCode/multilingual-ner</strong>

    </div>
    """,
    unsafe_allow_html=True,
)
