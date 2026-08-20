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

# ------------------------------------------------------------
# Replace these values with the exact information from your
# project before the final presentation.
# ------------------------------------------------------------

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background-color: #f7f8fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Typography ---------- */

    .hero-title {
        font-size: 3rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        line-height: 1.05;
        margin-bottom: 0.7rem;
        color: #111827;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #5b6472;
        max-width: 800px;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: #667085;
        line-height: 1.7;
        margin-bottom: 1.4rem;
    }

    /* ---------- Cards ---------- */

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
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #667085;
        font-weight: 650;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.25rem;
    }

    /* ---------- Entity highlighting ---------- */

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

    /* ---------- Pipeline ---------- */

    .pipeline-step {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        min-height: 125px;
    }

    .pipeline-number {
        font-size: 0.75rem;
        color: #667085;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .pipeline-title {
        font-weight: 700;
        color: #111827;
        margin-top: 0.3rem;
    }

    .pipeline-description {
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.4rem;
    }

    /* ---------- Notice ---------- */

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

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
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

    Streamlit reruns the application script when users interact
    with widgets. cache_resource prevents the model from being
    reconstructed during normal reruns.
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

    output.append(html.escape(text[cursor:]))

    return "".join(output)


def confidence_level(score: float) -> str:
    """Human-readable confidence interpretation."""

    if score >= 0.90:
        return "High"

    if score >= 0.70:
        return "Moderate"

    return "Lower"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ◈ Multilingual KYC NER")

    st.caption(
        "Named entity recognition for identity-related text."
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "NER Analysis",
            "Methodology",
            "Dataset",
            "Limitations",
            "Future Improvements",
            "System Architecture",
        ],
    )

    st.divider()

    st.markdown("**Model**")
    st.caption(MODEL_ID)

    st.markdown("**Task**")
    st.caption("Token classification / NER")

    st.divider()

    st.caption(
        "Research and demonstration interface"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">Multilingual KYC NER</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="hero-subtitle">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

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
        The interface separates inference from methodological
        documentation so that model behavior, training decisions,
        limitations, and possible extensions can be examined
        independently.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="metric-label">Task</div>
                <div class="metric-value">NER</div>
                <p>Token-level identification of relevant entities.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="metric-label">Domain</div>
                <div class="metric-value">KYC</div>
                <p>Identity-related information extraction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">Multilingual</div>
                <p>Transformer-based token classification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="info-card">
                <div class="metric-label">Output</div>
                <div class="metric-value">Entities</div>
                <p>Recognized spans with confidence scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

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
                    <div class="pipeline-number">{number}</div>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-description">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    st.markdown(
        """
        <div class="notice">
        <strong>Interpretation principle.</strong>
        The model identifies candidate entities in text. Entity
        recognition should not be interpreted as independent
        verification of the authenticity or truthfulness of the
        underlying identity information.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NER ANALYSIS
# ============================================================

elif page == "NER Analysis":

    st.markdown(
        '<div class="section-title">Named Entity Recognition</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Enter identity-related text below. The model will identify
        entity spans and provide confidence estimates for its
        predictions.
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
    )

    analyze = st.button(
        "Analyze Text",
        type="primary",
        use_container_width=True,
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
                        '<div class="section-title">Detected Entities</div>',
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
                        '<div class="section-title">Annotated Text</div>',
                        unsafe_allow_html=True,
                    )

                    highlighted = highlight_entities(
                        text,
                        results,
                    )

                    st.markdown(
                        f'<div class="entity-text">{highlighted}</div>',
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
                    st.code(str(exc))


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
        The major methodological decisions are presented separately
        so that each step can be evaluated for its purpose, benefit,
        and associated constraint.
        </div>
        """,
        unsafe_allow_html=True,
    )

    methodology = [
        (
            "1. Dataset Selection",
            "Relevant annotated NER datasets were selected according "
            "to their entity coverage, language coverage, annotation "
            "quality, and relevance to the target task."
        ),
        (
            "2. Dataset Analysis",
            "The datasets were examined for label definitions, "
            "annotation conventions, language distribution, and "
            "differences in data characteristics."
        ),
        (
            "3. Label Alignment",
            "Labels from different sources must represent a compatible "
            "entity schema before examples are combined. This reduces "
            "the risk of contradictory supervision."
        ),
        (
            "4. Dataset Merging",
            "Compatible training examples can be combined to increase "
            "the diversity and coverage of the training corpus."
        ),
        (
            "5. Data Splitting",
            "The resulting corpus is divided into training, validation, "
            "and test partitions so that model development and final "
            "evaluation remain conceptually separate."
        ),
        (
            "6. Tokenization",
            "Text is converted into the token representation required "
            "by the transformer model while preserving the relationship "
            "between tokens and entity labels."
        ),
        (
            "7. Fine-tuning",
            "A pretrained multilingual transformer is adapted to the "
            "NER task using the prepared labeled corpus."
        ),
        (
            "8. Evaluation",
            "Model predictions should be evaluated using entity-level "
            "precision, recall, and F1, preferably broken down by "
            "language and entity type."
        ),
    ]

    for title, description in methodology:

        with st.expander(title, expanded=False):

            st.write(description)

            if title == "4. Dataset Merging":

                st.markdown(
                    """
                    **Why merging matters**

                    A merged corpus can provide broader linguistic
                    and entity coverage than an individual dataset.
                    This may improve the model's ability to generalize
                    across different inputs.

                    **What must be controlled**

                    Merging is not automatically beneficial. Differences
                    in label definitions, annotation quality, language
                    distribution, and domain can introduce noise or
                    inconsistent supervision.
                    """
                )

    st.markdown(
        '<div class="section-title">Why This Pipeline?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        The pipeline separates data preparation from model inference.
        This is important because errors introduced during dataset
        construction or label alignment can propagate into training
        and affect downstream model performance. Treating each stage
        as an explicit component makes the system easier to evaluate,
        reproduce, and improve.
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
        The training data determines what the model can learn. The
        dataset documentation therefore records not only the sources,
        but also the decisions made when combining and standardizing
        them.
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
                This section intentionally avoids inventing those
                values.
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
                of examples available during training. The benefit
                depends on consistent entity definitions and annotation
                quality.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

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
        """
        ### Dataset merging: what matters

        Before merging datasets, the following questions should be
        answered:

        1. Do equivalent labels have equivalent meanings?
        2. Are annotation boundaries defined consistently?
        3. Are the same entity types represented across sources?
        4. Does one dataset dominate the merged distribution?
        5. Are language-specific differences being preserved?
        6. Are duplicate or near-duplicate examples present?
        7. Are licensing conditions compatible?

        A merged dataset is useful only when the resulting supervision
        remains coherent.
        """
    )


# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.markdown(
        '<div class="section-title">Constraints & Limitations</div>',
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
            "conditions encountered in deployment."
        ),
        (
            "Annotation consistency",
            "Combining datasets with different annotation conventions can "
            "introduce inconsistent supervision even when labels have "
            "similar names."
        ),
        (
            "Domain shift",
            "Performance may decrease when deployment text differs "
            "substantially from the training data."
        ),
        (
            "Document quality",
            "If text is obtained through OCR, recognition errors may be "
            "propagated into the NER stage."
        ),
        (
            "Confidence interpretation",
            "Model confidence is not equivalent to factual verification. "
            "A highly confident prediction can still be incorrect."
        ),
        (
            "KYC verification",
            "Entity extraction alone does not establish that a person's "
            "identity or document is genuine."
        ),
        (
            "Resource requirements",
            "The transformer model is relatively large and therefore "
            "requires substantially more memory than a lightweight "
            "rule-based extractor."
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
        '<div class="section-title">Potential Improvements</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Future development should target measurable weaknesses rather
        than adding functionality solely for presentation. The
        following improvements are organized according to the part of
        the system they would affect.
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
                st.markdown(f"- {item}")


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
        The application separates presentation, model loading, and
        inference concerns. This allows the interface and supporting
        documentation to evolve without changing the trained model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture = [
        (
            "User Interface",
            "Streamlit provides the interaction layer for text input, "
            "results, visualization, and documentation."
        ),
        (
            "Inference Layer",
            "The application sends input text to the NER pipeline and "
            "receives entity spans and confidence scores."
        ),
        (
            "Tokenizer",
            "The tokenizer converts text into the representation expected "
            "by the transformer model."
        ),
        (
            "Multilingual NER Model",
            MODEL_ID
            + " provides the fine-tuned token classification model."
        ),
        (
            "Output Layer",
            "Predictions are transformed into human-readable entity "
            "tables and annotated text."
        ),
    ]

    for index, (title, description) in enumerate(architecture, 1):

        col1, col2 = st.columns([1, 5])

        with col1:

            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-number">
                        COMPONENT {index}
                    </div>
                    <div class="pipeline-title">
                        {title}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if index != len(architecture):
            st.write("")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multilingual KYC NER • Research and demonstration interface"
)

st.caption(
    "Model: SoloCode/multilingual-ner"
)
