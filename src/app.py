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
    initial_sidebar_state="expanded",
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
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "NER Analysis"

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
        background-color: #f7f8fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* ========================================================
       BRAND BAR
       ======================================================== */

    .brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-icon {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: #eef2ff;
        color: #4f46e5;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .brand-name {
        font-weight: 750;
        color: #111827;
        font-size: 1rem;
    }

    .brand-caption {
        color: #667085;
        font-size: 0.78rem;
        margin-top: 0.1rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        display: inline-block;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 1rem 0 2rem 0;
    }

    .hero-eyebrow {
        color: #4f46e5;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        line-height: 1.05;
        color: #111827;
        margin-bottom: 0.7rem;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.7;
        color: #5b6472;
        max-width: 820px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #4338ca;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ========================================================
       SECTIONS
       ======================================================== */

    .section-title {
        font-size: 1.8rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1.6rem;
        margin-bottom: 0.45rem;
    }

    .section-description {
        color: #667085;
        line-height: 1.7;
        margin-bottom: 1.4rem;
        max-width: 850px;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }

    .info-card h3 {
        margin-top: 0;
        font-size: 1.05rem;
        color: #111827;
    }

    .info-card p {
        color: #667085;
        line-height: 1.65;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #667085;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 750;
        color: #111827;
        margin-top: 0.25rem;
    }

    /* ========================================================
       ENTITY HIGHLIGHTING
       ======================================================== */

    .entity-text {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.4rem;
        line-height: 2.1;
        font-size: 1rem;
        color: #1f2937;
    }

    .entity {
        display: inline-block;
        padding: 0.12rem 0.42rem;
        margin: 0 0.08rem;
        border-radius: 6px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
    }

    .entity-label {
        font-size: 0.68rem;
        font-weight: 750;
        color: #4338ca;
        margin-left: 0.3rem;
        vertical-align: middle;
    }

    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline-step {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        min-height: 125px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }

    .pipeline-number {
        font-size: 0.72rem;
        color: #667085;
        font-weight: 750;
        letter-spacing: 0.08em;
    }

    .pipeline-title {
        font-weight: 750;
        color: #111827;
        margin-top: 0.3rem;
    }

    .pipeline-description {
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.4rem;
    }

    /* ========================================================
       NOTICES
       ======================================================== */

    .notice {
        border-left: 4px solid #4f46e5;
        background: #eef2ff;
        padding: 1rem 1.1rem;
        border-radius: 0 10px 10px 0;
        color: #3730a3;
        line-height: 1.6;
    }

    .warning {
        border-left: 4px solid #d97706;
        background: #fffbeb;
        padding: 1rem 1.1rem;
        border-radius: 0 10px 10px 0;
        color: #92400e;
        line-height: 1.6;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #667085;
        font-weight: 800;
        margin-bottom: 0.5rem;
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
    Load the Hugging Face tokenizer and model once.

    Streamlit reruns the script when widgets change, but
    cache_resource prevents the model from being reconstructed
    during normal reruns.
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
    """Safely convert a confidence value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_entity_label(label: str) -> str:
    """Convert BIO-style labels into readable labels."""

    if not label:
        return "ENTITY"

    label = label.replace("B-", "")
    label = label.replace("I-", "")

    return label


def highlight_entities(
    text: str,
    entities: list[dict],
) -> str:
    """
    Highlight detected entities using character offsets.
    """

    if not entities:
        return html.escape(text)

    sorted_entities = sorted(
        entities,
        key=lambda item: item.get("start", 0),
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
                entity.get(
                    "entity_group",
                    "ENTITY",
                )
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
    """Human-readable confidence interpretation."""

    if score >= 0.90:
        return "High"

    if score >= 0.70:
        return "Moderate"

    return "Lower"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.05rem;
            font-weight:800;
            color:#111827;
            margin-bottom:0.2rem;
        ">
            ◈ Multilingual KYC NER
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Research and demonstration system"
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section-title">System</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "◉  NER Analysis",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.page == "NER Analysis"
            else "secondary"
        ),
    ):
        st.session_state.page = "NER Analysis"
        st.rerun()

    st.markdown(
        '<div class="sidebar-section-title">Documentation</div>',
        unsafe_allow_html=True,
    )

    documentation_pages = [
        ("Overview", "Overview"),
        ("Methodology", "Methodology"),
        ("Dataset", "Dataset"),
        ("Constraints & Limitations", "Limitations"),
        ("Future Improvements", "Future Improvements"),
        ("System Architecture", "System Architecture"),
    ]

    for label, value in documentation_pages:

        if st.button(
            label,
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == value
                else "secondary"
            ),
            key=f"nav_{value}",
        ):
            st.session_state.page = value
            st.rerun()

    st.divider()

    st.markdown(
        "**Model**"
    )

    st.caption(
        MODEL_ID
    )

    st.markdown(
        "**Task**"
    )

    st.caption(
        "Token classification / NER"
    )

    st.divider()

    st.caption(
        "INSA Summer Camp"
    )


page = st.session_state.page


# ============================================================
# TOP BRAND BAR
# ============================================================

st.markdown(
    """
    <div class="brand-bar">

        <div class="brand-left">

            <div class="brand-icon">
                ◈
            </div>

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
            {html.escape(APP_NAME)}
        </div>

        <div class="hero-subtitle">
            {html.escape(APP_SUBTITLE)}
        </div>

        <div class="hero-badge">
            Transformer-based Named Entity Recognition
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NER ANALYSIS
# DEFAULT PAGE
# ============================================================

if page == "NER Analysis":

    st.markdown(
        '<div class="section-title">Named Entity Recognition</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Enter identity-related text below. The trained model
            identifies candidate entity spans and provides a
            confidence score for each prediction.
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
        help=(
            "Enter identity-related text for the NER model "
            "to analyze."
        ),
    )

    st.caption(
        "Example format: Full Name, Date of Birth, "
        "Passport Number, and other identity-related fields."
    )

    analyze = st.button(
        "Analyze Text",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="warning" style="margin-top:1rem;">
            <strong>Important:</strong>
            NER identifies candidate entities in text. It does not
            independently verify the authenticity of an identity,
            document, or piece of information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analyze:

        if not text.strip():

            st.warning(
                "Please enter text before starting the analysis."
            )

        else:

            try:

                with st.spinner(
                    "Running multilingual NER inference..."
                ):

                    ner = load_ner_pipeline()
                    results = ner(text)

                st.success(
                    f"Analysis completed. "
                    f"{len(results)} entity span(s) detected."
                )

                if results:

                    st.markdown(
                        '<div class="section-title">'
                        'Detected Entities'
                        '</div>',
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
                                        "ENTITY",
                                    )
                                ),
                                "Text": item.get(
                                    "word",
                                    "",
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
                        f"""
                        <div class="entity-text">
                            {highlighted}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Confidence represents the model's prediction "
                        "score for the detected span. It should not be "
                        "interpreted as a probability that the identity "
                        "information is factually correct."
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


# ============================================================
# OVERVIEW
# ============================================================

elif page == "Overview":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            This application demonstrates a multilingual named entity
            recognition system designed to identify structured
            identity-related information from unstructured KYC text.
            The interface separates model inference from methodological
            documentation so that the system's training decisions,
            constraints, and potential improvements can be examined
            independently.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        (
            "Task",
            "NER",
            "Token-level identification of relevant entities.",
        ),
        (
            "Domain",
            "KYC",
            "Identity-related information extraction.",
        ),
        (
            "Model",
            "Multilingual",
            "Transformer-based token classification.",
        ),
        (
            "Output",
            "Entities",
            "Recognized spans with confidence scores.",
        ),
    ]

    for column, (label, value, description) in zip(
        [col1, col2, col3, col4],
        cards,
    ):

        with column:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="metric-label">
                        {html.escape(label)}
                    </div>

                    <div class="metric-value">
                        {html.escape(value)}
                    </div>

                    <p>
                        {html.escape(description)}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Processing Pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "01",
            "Input",
            "KYC text is provided to the system.",
        ),
        (
            "02",
            "Preprocessing",
            "Input is normalized where required.",
        ),
        (
            "03",
            "Tokenization",
            "Text is converted into model-compatible tokens.",
        ),
        (
            "04",
            "NER Inference",
            "The fine-tuned transformer predicts entity spans.",
        ),
        (
            "05",
            "Aggregation",
            "Token predictions are combined into entities.",
        ),
        (
            "06",
            "Presentation",
            "Entities and confidence scores are displayed.",
        ),
    ]

    cols = st.columns(3)

    for index, (number, title, description) in enumerate(steps):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="pipeline-step">

                    <div class="pipeline-number">
                        {html.escape(number)}
                    </div>

                    <div class="pipeline-title">
                        {html.escape(title)}
                    </div>

                    <div class="pipeline-description">
                        {html.escape(description)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="notice" style="margin-top:1.2rem;">
            <strong>Interpretation principle.</strong>
            The model identifies candidate entities in text.
            Entity recognition should not be interpreted as independent
            verification of the authenticity or truthfulness of the
            underlying identity information.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="section-title">Methodology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The system follows a supervised multilingual NER workflow.
            Each major methodological decision is documented according
            to its purpose, benefit, and associated constraint.
        </div>
        """,
        unsafe_allow_html=True,
    )

    methodology = [
        (
            "1. Dataset Selection",
            "Relevant annotated NER datasets were selected according "
            "to their entity coverage, language coverage, annotation "
            "quality, and relevance to the target task.",
        ),
        (
            "2. Dataset Analysis",
            "The datasets were examined for label definitions, "
            "annotation conventions, language distribution, and "
            "differences in data characteristics.",
        ),
        (
            "3. Label Alignment",
            "Labels from different sources must represent a compatible "
            "entity schema before examples are combined. This reduces "
            "the risk of contradictory supervision.",
        ),
        (
            "4. Dataset Merging",
            "Compatible training examples can be combined to increase "
            "the diversity and coverage of the training corpus.",
        ),
        (
            "5. Data Splitting",
            "The resulting corpus is divided into training, validation, "
            "and test partitions so that model development and final "
            "evaluation remain conceptually separate.",
        ),
        (
            "6. Tokenization",
            "Text is converted into the token representation required "
            "by the transformer model while preserving the relationship "
            "between tokens and entity labels.",
        ),
        (
            "7. Fine-tuning",
            "A pretrained multilingual transformer is adapted to the "
            "NER task using the prepared labeled corpus.",
        ),
        (
            "8. Evaluation",
            "Model predictions should be evaluated using entity-level "
            "precision, recall, and F1, preferably broken down by "
            "language and entity type.",
        ),
    ]

    for title, description in methodology:

        with st.expander(title):

            st.write(description)

            if title == "4. Dataset Merging":

                st.markdown(
                    """
                    ### Why merging matters

                    A merged corpus can provide broader linguistic and
                    entity coverage than an individual dataset. This can
                    give the model exposure to a wider range of examples.

                    ### What must be controlled

                    Merging is not automatically beneficial. Differences
                    in label definitions, annotation quality, language
                    distribution, and domain can introduce noise or
                    inconsistent supervision.

                    ### Key principle

                    Datasets should be merged only after their annotation
                    schemes and entity definitions have been examined for
                    compatibility.
                    """
                )

    st.markdown(
        '<div class="section-title">Why This Pipeline?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">

        <p>
        The pipeline separates data preparation from model inference.
        This is important because errors introduced during dataset
        construction or label alignment can propagate into training
        and affect downstream model performance.
        </p>

        <p>
        Treating each stage as an explicit component makes the system
        easier to evaluate, reproduce, explain, and improve.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The training data determines what the model can learn.
            Dataset documentation therefore records the sources,
            standardization decisions, merging strategy, and constraints
            associated with the resulting training corpus.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">

                <h3>Dataset Sources</h3>

                <p>
                The exact dataset names, versions, licenses, language
                coverage, and sample counts should be documented here.
                These values should correspond to the actual training
                notebooks and dataset preparation process.
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
                Combining compatible datasets can increase the diversity
                of examples available during training. The benefit,
                however, depends on consistent entity definitions,
                annotation quality, and language coverage.
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
        '<div class="section-title">'
        'Dataset Merging: What Matters'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">

        <p>
        Before merging datasets, the following questions should be
        answered:
        </p>

        <ol>
            <li>Do equivalent labels have equivalent meanings?</li>
            <li>Are annotation boundaries defined consistently?</li>
            <li>Are the same entity types represented across sources?</li>
            <li>Does one dataset dominate the merged distribution?</li>
            <li>Are language-specific differences being preserved?</li>
            <li>Are duplicate or near-duplicate examples present?</li>
            <li>Are licensing conditions compatible?</li>
        </ol>

        <p>
        A merged dataset is useful only when the resulting supervision
        remains coherent.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.markdown(
        '<div class="section-title">'
        'Constraints & Limitations'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            A reliable ML system should document where its assumptions
            stop. The following limitations should be considered when
            interpreting predictions.
        </div>
        """,
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

        with st.expander(title):

            st.write(description)


# ============================================================
# FUTURE IMPROVEMENTS
# ============================================================

elif page == "Future Improvements":

    st.markdown(
        '<div class="section-title">'
        'Potential Improvements'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Future development should target measurable weaknesses
            rather than adding functionality solely for presentation.
            The following improvements are organized according to
            the part of the system they would affect.
        </div>
        """,
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

    for category, items in improvements.items():

        with st.expander(category, expanded=True):

            for item in items:
                st.markdown(
                    f"- {item}"
                )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

elif page == "System Architecture":

    st.markdown(
        '<div class="section-title">'
        'System Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            The application separates presentation, model loading,
            and inference concerns. This allows the interface and
            supporting documentation to evolve without changing the
            trained model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture = [
        (
            "User Interface",
            "Streamlit provides the interaction layer for text input, "
            "results, visualization, and documentation.",
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

        col1, col2 = st.columns([1, 5])

        with col1:

            st.markdown(
                f"""
                <div class="pipeline-step">

                    <div class="pipeline-number">
                        COMPONENT {index}
                    </div>

                    <div class="pipeline-title">
                        {html.escape(title)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card">

                    <p>
                        {html.escape(description)}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#667085;
        font-size:0.82rem;
        line-height:1.7;
    ">

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
