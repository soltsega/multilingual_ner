import streamlit as st
from transformers import pipeline

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "SoloCode/multilingual-ner"


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_ner_model():

    ner_pipeline = pipeline(
        "token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple"
    )

    return ner_pipeline


ner_pipeline = load_ner_model()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Multilingual NER",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Multilingual Named Entity Recognition")

st.write(
    "Enter a sentence and the trained model will identify "
    "named entities."
)

text = st.text_area(
    "Enter a sentence:",
    placeholder="Example: Nelson Mandela was born in South Africa."
)

if st.button("Recognize Entities"):

    if not text.strip():

        st.warning("Please enter a sentence.")

    else:

        results = ner_pipeline(text)

        if not results:

            st.info("No named entities detected.")

        else:

            st.subheader("Recognized Entities")

            for entity in results:

                st.write(
                    f"**{entity['word']}** → "
                    f"`{entity['entity_group']}` "
                    f"({entity['score']:.2%})"
                )