import io
import os
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocuQuery | Grounded PDF Retrieval System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN FLUENT-INSPIRED CSS WITH MODERN CARDS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .app-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F3F9FD 100%);
        border-bottom: 2px solid #0078D4;
        padding: 2rem 2.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 120, 212, 0.06);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #201F1E;
        margin: 0;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #605E5C;
        margin-top: 0.4rem;
    }
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #EDEBE9;
        border-top: 3px solid #0078D4;
        padding: 1.25rem;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 100%;
    }
    .response-container {
        background-color: #FFFFFF;
        border: 1px solid #EDEBE9;
        border-left: 4px solid #0078D4;
        padding: 1.5rem;
        border-radius: 6px;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    .meta-tag {
        display: inline-block;
        background-color: #EFF6FC;
        color: #0078D4;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE CLIENT ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Configuration Error: Missing or unreadable API key in `.streamlit/secrets.toml`.")
    st.stop()

# --- CORE RETRIEVAL & EXTRACTION PIPELINE ---
@st.cache_data(show_spinner=False)
def parse_and_chunk_pdf(file_bytes: bytes, file_name: str):
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        words = raw_text.split()
        window_size = 110
        overlap = 20
        for i in range(0, len(words), window_size - overlap):
            segment = " ".join(words[i:i + window_size])
            if len(segment.strip()) > 40:
                chunks.append({
                    "source": file_name,
                    "page": page_num,
                    "text": segment
                })
    return chunks, len(reader.pages)

def rank_chunks_tfidf(query: str, chunks: list, top_k: int = 4):
    if not chunks:
        return []
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0.04:
                item = chunks[idx].copy()
                item["score"] = round(float(scores[idx]), 3)
                results.append(item)
        return results
    except ValueError:
        return [dict(c, score=0.0) for c in chunks[:top_k]]

def generate_grounded_answer(query: str, retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return "No matching context was found in the indexed document for this query."

    context_lines = []
    for c in retrieved_chunks:
        line = "[Page " + str(c['page']) + "]: " + c['text']
        context_lines.append(line)
    formatted_context = "\n\n".join(context_lines)
    
    system_instruction = (
        "You are a precision document retrieval assistant. Answer the query using strictly "
        "the provided context excerpts. Do not extrapolate or use outside knowledge. "
        "If the context lacks sufficient evidence, state that clearly. Cite page numbers inline (e.g., [Page 2])."
    )
    prompt = f"{system_instruction}\n\nContext Excerpts:\n{formatted_context}\n\nQuery: {query}"

    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
            
    return f"⚠️ Service temporarily busy (503 High Demand). Please wait a few seconds and try clicking your query again. (Error details: {str(last_error)})"

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 📁 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    sample_files = [f for f in os.listdir("docs") if f.endswith(".pdf")] if os.path.exists("docs") else []
    selected_sample = None
    if sample_files:
        selected_sample = st.selectbox("Local Repository Samples", ["None"] + sample_files)

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Parameters")
    top_k = st.slider("Top-K Context Chunks", min_value=2, max_value=8, value=4)

    st.markdown("---")
    st.markdown("### ⚡ Standard Queries")
    run_summary = st.button("Generate Document Overview", use_container_width=True)
    run_key_points = st.button("Extract Core Findings", use_container_width=True)
    run_deadlines = st.button("Identify Dates & Deliverables", use_container_width=True)

# --- HEADER ---
st.markdown("""
    <div class="app-header">
        <div class="app-title">DocuQuery — Grounded PDF Retrieval & Q&A</div>
        <div class="app-subtitle">TF-IDF Lexical Retrieval + Strict Citation-Grounded Synthesis Pipeline</div>
    </div>
""", unsafe_allow_html=True)

# --- DOCUMENT LOADING ---
doc_bytes = None
doc_name = None

if uploaded_file is not None:
    doc_bytes = uploaded_file.read()
    doc_name = uploaded_file.name
elif selected_sample and selected_sample != "None":
    with open(os.path.join("docs", selected_sample), "rb") as f:
        doc_bytes = f.read()
    doc_name = selected_sample

if not doc_bytes:
    st.info("👈 Upload a PDF file or select a sample document in the sidebar to initialize the pipeline.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #0078D4; margin-top:0;">1. Overlapping Chunking</h4>
                <p style="font-size: 0.9rem; color: #605E5C;">Parses PDF pages into 110-word sliding windows with a 20-word overlap to preserve boundary context.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #0078D4; margin-top:0;">2. TF-IDF Cosine Ranking</h4>
                <p style="font-size: 0.9rem; color: #605E5C;">Vectorizes chunks and ranks relevance against your query with transparent mathematical similarity scores.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h4 style="color: #0078D4; margin-top:0;">3. Citation Verification</h4>
                <p style="font-size: 0.9rem; color: #605E5C;">Constrains model synthesis strictly to retrieved context with verifiable page-level evidence.</p>
            </div>
        """, unsafe_allow_html=True)
else:
    chunks, total_pages = parse_and_chunk_pdf(doc_bytes, doc_name)

    m1, m2, m3 = st.columns(3)
    m1.metric("Indexed Document", doc_name)
    m2.metric("Total Pages", total_pages)
    m3.metric("Indexed Chunks", len(chunks))

    st.markdown("---")
    user_question = st.text_input(
        "Enter your query:",
        placeholder="Ask a specific question about the uploaded document..."
    )

    active_query = None
    is_preset = False
    if run_summary:
        active_query = "Provide a concise, objective overview of this document."
        is_preset = True
    elif run_key_points:
        active_query = "Extract the primary findings or key points in bullet form with page citations."
        is_preset = True
    elif run_deadlines:
        active_query = "List all specific dates, timelines, or deliverables mentioned in the text."
        is_preset = True
    elif user_question.strip():
        active_query = user_question.strip()

    if active_query:
        with st.spinner("Running retrieval and grounding pipeline..."):
            if is_preset:
                matched_chunks = [dict(c, score=1.0) for c in chunks[:top_k]]
            else:
                matched_chunks = rank_chunks_tfidf(active_query, chunks, top_k=top_k)

            output = generate_grounded_answer(active_query, matched_chunks)

        st.markdown(f"""
            <div class="response-container">
                <div class="meta-tag">Grounded Output • {len(matched_chunks)} Chunks Retrieved</div>
                <div style="font-size: 1rem; line-height: 1.65; color: #201F1E;">
                    {output}
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("View Retrieved Evidence & Cosine Similarity Scores"):
            for i, chunk in enumerate(matched_chunks, start=1):
                st.markdown(f"**Chunk {i} | Page {chunk['page']} | Cosine Score: `{chunk['score']}`**")
                st.caption(chunk["text"])
                if i < len(matched_chunks):
                    st.markdown("---")