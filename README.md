# NER-KYC

This project aims to build a Named Entity Recognition (NER) pipeline for Know Your Customer (KYC) documents. The goal is to automate the extraction of key entities (like names, dates of birth, addresses, and ID numbers) from documents such as passports, national IDs, and bank statements.

## Project Structure

- `data/`: Raw and processed dataset files.
- `doc/`: Project documentation and notes.
- `models/`: Saved models and weights.
- `notebooks/`: Jupyter notebooks for data exploration and model experiments.
- `src/`: Main source code for OCR, preprocessing, and NER.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ner-kyc
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Place your raw KYC documents in the `data/raw/` directory.

## Pipeline Overview

1. **OCR**: Convert images/scanned PDFs to raw text.
2. **Preprocessing**: Clean the text and fix layout issues.
3. **NER**: Extract key entities using a pre-trained/fine-tuned model.
4. **Validation**: Validate the extracted fields and structure as JSON.
