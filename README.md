# PDF Question-Answering Assistant

A retrieval-based PDF exploration app. Users upload a text-based PDF, ask a question, and receive the most relevant passages with page numbers.

## Features
- Extract text from PDFs with pypdf
- Split text into overlapping passages
- Rank passages with TF-IDF and cosine similarity
- Display page references and similarity scores

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Scope and limitations
This starter performs retrieval, not generative question answering. TF-IDF can miss semantic matches. Scanned PDFs require OCR, which is not included. Do not upload confidential documents to a hosted version without appropriate privacy safeguards.

## Skills demonstrated
Python, PDF parsing, text chunking, information retrieval, TF-IDF, cosine similarity, and Streamlit.
