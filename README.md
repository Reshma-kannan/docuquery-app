# DocuQuery — Grounded PDF Retrieval & Q&A System

A production-grade, citation-backed Retrieval-Augmented Generation (RAG) application engineered for enterprise document intelligence. Designed with Microsoft Fluent Design principles, this system provides transparent lexical retrieval, sliding-window chunking, and strict grounding constraints to eliminate hallucinations.

---

## 🏗️ Architectural Overview

Unlike black-box LLM wrappers, **DocuQuery** prioritizes engineering rigor and full transparency by exposing the underlying vector scoring and chunking mechanics:

1. **Document Ingestion & Chunking (`pypdf`)**: 
   - Parses uploaded PDF documents page by page.
   - Applies a **sliding-window chunking algorithm** (110-word windows with a 20-word overlap) to preserve semantic context across page boundaries.
2. **Lexical Vectorization & Scoring (`scikit-learn`)**: 
   - Utilizes `TfidfVectorizer` (with English stop-word removal) to build a sparse term-frequency matrix.
   - Computes **cosine similarity** between the user query and document chunks to rank and retrieve the top-$K$ most relevant excerpts with precise mathematical scores.
3. **Grounded Synthesis (`Google GenAI SDK`)**: 
   - Feeds retrieved context into an LLM pipeline with strict system instructions.
   - Enforces zero external knowledge usage and mandates inline page-level citations (e.g., `[Page 2]`).

---

## 💻 Tech Stack

* **Frontend / UI**: Streamlit (configured with Microsoft Fluent-inspired light theme and neutral typography).
* **PDF Parsing**: `pypdf`
* **Vectorization & Similarity**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
* **LLM Engine**: Google GenAI SDK (`gemini-3.8-flash` with multi-model fallback support)

---

## 🚀 Getting Started & Local Deployment

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Open your terminal inside the project directory and install the required packages:
```bash
pip install streamlit pypdf scikit-learn google-genai