import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
st.title("📄 PDF Question-Answering Assistant")
st.write("Ask a question and retrieve the most relevant passages from your PDF. This demo retrieves evidence; it does not generate answers with an LLM.")
file = st.file_uploader("Upload a text-based PDF", type=["pdf"])
def split_chunks(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
if file:
    try:
        reader = PdfReader(file)
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        chunks = []
        for page_num, text in pages:
            for chunk in split_chunks(text):
                chunks.append({"page": page_num, "text": chunk})
        if not chunks:
            st.warning("No extractable text found. This may be a scanned PDF; OCR is not included in this starter.")
        else:
            st.success(f"Extracted text into {len(chunks)} passages.")
            question = st.text_input("Your question")
            top_k = st.slider("Evidence passages", 1, min(5, len(chunks)), min(3, len(chunks)))
            if st.button("Find evidence", type="primary", disabled=not question.strip()):
                corpus = [c["text"] for c in chunks]
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                matrix = vectorizer.fit_transform(corpus + [question])
                scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
                indices = scores.argsort()[::-1][:top_k]
                st.subheader("Relevant passages")
                for rank, idx in enumerate(indices, 1):
                    c = chunks[int(idx)]
                    st.markdown(f"**{rank}. Page {c['page']} · Similarity {scores[idx]:.3f}**")
                    st.write(c["text"])
                    st.divider()
                st.caption("Similarity is lexical TF-IDF similarity, not proof that a passage answers the question.")
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
