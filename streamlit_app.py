import os
import streamlit as st

# Make Streamlit secrets visible as env vars BEFORE importing project code
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

st.set_page_config(page_title="Infosys Annual Report - GraphRAG", layout="wide")
st.title("Organizational RAG - Infosys Annual Report")

@st.cache_resource(show_spinner="Loading models and connecting to databases (first load takes a minute)...")
def load_backend():
    from src.retrieval import answer_with_sources
    from src.baseline_rag import baseline_answer
    from src.ingest import ingest_text
    return answer_with_sources, baseline_answer, ingest_text

answer_with_sources, baseline_answer, ingest_text = load_backend()

def pdf_to_text(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)

tab1, tab2 = st.tabs(["Ask a question", "Upload a document"])

with tab1:
    q = st.text_input("Question")
    mode = st.radio("Mode", ["GraphRAG", "Baseline"], horizontal=True)
    if st.button("Ask") and q:
        with st.spinner("Thinking..."):
            if mode == "GraphRAG":
                answer, chunks = answer_with_sources(q)
            else:
                answer, chunks = baseline_answer(q)
        st.write(answer)
        with st.expander("Sources used"):
            for c in chunks:
                st.caption(c[:300])

with tab2:
    required = os.environ.get("UPLOAD_PASSWORD", "")
    entered = st.text_input("Upload password", type="password") if required else ""
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Ingest"):
        if required and entered != required:
            st.error("Wrong or missing upload password.")
        else:
            with st.spinner("Parsing and indexing (can take a few minutes)..."):
                text = pdf_to_text(uploaded)
                if not text.strip():
                    st.error("No extractable text found (scanned PDF?).")
                else:
                    st.success(ingest_text(text, uploaded.name))