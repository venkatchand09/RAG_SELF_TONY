# Advanced RAG Streamlit app for biotech/microbe documents
# Features:
# - Uses PyMuPDF (fitz) to extract text + images from PDFs (falls back to pdfminer if needed)
# - Optionally performs OCR on embedded images (pytesseract) to capture text inside figures/tables
# - Uses ChromaDB for an efficient, persistent vector store
# - Uses OpenAI embeddings (configurable model) and Chat completions for answers
# - Conversational memory stored in session_state
# - Produces answers in the voice of a biotech/medical researcher, with hypothesis generation,
#   follow-up suggestions and source citations under each answer
# - UI for indexing, monitoring progress, search, and downloading context/hypotheses

import streamlit as st
import fitz  # PyMuPDF
import os
import io
import json
import time
import base64
import tempfile
import math
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ChromaDB (pip install chromadb)
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# Simple sentence splitter / chunker (keeps dependency light)
import re

# ----------------------------- Utility functions -----------------------------

def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def simple_chunk_text(text: str, chunk_size: int = 800, overlap: int = 128) -> List[str]:
    """Chunk text into pieces of roughly chunk_size tokens (approx by characters) with overlap."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # try to cut at sentence boundary
        if end < len(text):
            m = re.search(r"[\.\?\!][ \n]", text[end - 50 : end + 50])
            if m:
                rel = m.end() - 50
                end = end - 50 + rel
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start += step
    return chunks


# ----------------------------- PDF Processing -----------------------------

def extract_from_pdf_bytes(file_bytes: bytes, filename: str = "<pdf>", do_ocr_images: bool = False) -> Dict[str, Any]:
    """Extract text, images and metadata from a PDF file using PyMuPDF.
    Returns a dict with content_text, images_ocr_text (if enabled), images (list), metadata
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []
    image_texts = []
    images_info = []

    metadata = doc.metadata or {}
    num_pages = doc.page_count

    for i in range(num_pages):
        page = doc.load_page(i)
        # Extract text
        page_text = page.get_text("text") or ""
        # Also try to get blocks for better order
        if not page_text:
            page_text = page.get_text("blocks")
            if isinstance(page_text, list):
                page_text = "\n".join([b[4] for b in page_text])
        full_text.append(page_text)

        # Extract images on page
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image.get("image")
            image_ext = base_image.get("ext", "png")
            images_info.append({
                "page": i + 1,
                "xref": xref,
                "ext": image_ext,
                "bytes": image_bytes,
            })
            if do_ocr_images and OCR_AVAILABLE:
                try:
                    pil = Image.open(io.BytesIO(image_bytes))
                    text_from_img = pytesseract.image_to_string(pil)
                    if text_from_img.strip():
                        image_texts.append({
                            "page": i + 1,
                            "text": text_from_img.strip(),
                            "ext": image_ext,
                        })
                except Exception:
                    pass

    content_text = "\n\n".join(full_text)
    return {
        "filename": filename,
        "metadata": metadata,
        "content_text": content_text,
        "images_info": images_info,
        "images_ocr_text": image_texts,
    }


# ----------------------------- Web extraction -----------------------------

def get_all_urls(base_url: str, max_urls: int = 200) -> List[str]:
    urls = set()
    try:
        r = requests.get(base_url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc == urlparse(base_url).netloc:
                cleaned = parsed.scheme + "://" + parsed.netloc + parsed.path
                urls.add(cleaned)
                if len(urls) >= max_urls:
                    break
    except Exception as e:
        st.warning(f"Crawl error for {base_url}: {e}")
    return list(urls)


def extract_text_from_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator=" \n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text
    except Exception as e:
        st.warning(f"Failed to fetch {url}: {e}")
        return None


# ----------------------------- Embeddings and Vector DB -----------------------------

# We'll use OpenAI embeddings by calling the embeddings endpoint with requests directly.
# The code below uses a function get_openai_embeddings which supports batching.


def get_openai_embeddings(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results = []
    # Batch to avoid huge payloads
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": model, "input": batch}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Embedding API error {r.status_code}: {r.text}")
        data = r.json()
        for obj in data.get("data", []):
            results.append(obj["embedding"])
        time.sleep(0.1)
    return results


def init_chroma(persist_directory: str = "./chroma_db"):
    if not CHROMA_AVAILABLE:
        raise RuntimeError("ChromaDB package not installed. Install with `pip install chromadb`.")
    make_dirs(persist_directory)
    client = chromadb.Client(Settings(chroma_db_impl="chromadb.db.sqlite", persist_directory=persist_directory))
    return client


# ----------------------------- RAG pipeline (index + query) -----------------------------

def index_documents(
    uploaded_files,
    web_urls: Optional[str],
    openai_api_key: str,
    persist_directory: str = "./chroma_db",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-large",
    do_ocr_images: bool = False,
    crawl_website: bool = False,
):
    """Process PDFs and websites, chunk text, create embeddings and store into ChromaDB.
    Returns chroma client and metadata about the indexed collection.
    """
    client = init_chroma(persist_directory)
    coll_name = "biotech_rag"
    # create or get collection
    try:
        coll = client.get_collection(name=coll_name)
    except Exception:
        coll = client.create_collection(name=coll_name, embedding_function=None)

    documents = []
    metadatas = []
    ids = []

    # Process PDFs
    if uploaded_files:
        for f in uploaded_files:
            try:
                raw = f.read()
                extracted = extract_from_pdf_bytes(raw, filename=f.name, do_ocr_images=do_ocr_images)
                text = extracted["content_text"]
                if extracted["images_ocr_text"]:
                    for it in extracted["images_ocr_text"]:
                        text += "\n\n[OCR from image page %d]:\n" % it["page"] + it["text"]
                chunks = simple_chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                for i, ch in enumerate(chunks):
                    documents.append(ch)
                    metadatas.append({
                        "source": "pdf",
                        "filename": f.name,
                        "page_chunk": i,
                    })
            except Exception as e:
                st.warning(f"Failed to process {f.name}: {e}")

    # Process websites
    urls_list = []
    if web_urls:
        urls = [u.strip() for u in web_urls.split(",") if u.strip()]
        if crawl_website:
            for base in urls:
                discovered = get_all_urls(base)
                urls_list.extend(discovered)
        else:
            urls_list.extend(urls)

        for url in urls_list:
            content = extract_text_from_html(url)
            if content:
                chunks = simple_chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
                for i, ch in enumerate(chunks):
                    documents.append(ch)
                    metadatas.append({"source": "web", "url": url, "page_chunk": i})

    if not documents:
        st.warning("No textual content found to index.")
        return None

    st.info(f"Creating {len(documents)} chunks and generating embeddings (model={embedding_model})")
    embeddings = get_openai_embeddings(documents, api_key=openai_api_key, model=embedding_model)

    # Chroma expects a vector embedding function or we can insert with list of embeddings
    # We'll upsert manually by building a temporary collection
    # To ensure id uniqueness, use a prefix + index
    for i, (d, m, emb) in enumerate(zip(documents, metadatas, embeddings)):
        uid = f"doc_{int(time.time())}_{i}"
        try:
            coll.add(documents=[d], metadatas=[m], ids=[uid], embeddings=[emb])
        except Exception as e:
            # fallback: try to recreate collection
            try:
                client.delete_collection(coll_name)
            except Exception:
                pass
            coll = client.create_collection(name=coll_name, embedding_function=None)
            coll.add(documents=[d], metadatas=[m], ids=[uid], embeddings=[emb])

    client.persist()
    st.success("Indexing complete and saved to ChromaDB.")
    return {
        "client": client,
        "collection_name": coll_name,
        "num_chunks": len(documents),
        "persist_dir": persist_directory,
        "embedding_model": embedding_model,
    }


# ----------------------------- Querying + Answer Generation -----------------------------

def chroma_query_and_answer(
    client,
    collection_name: str,
    query: str,
    openai_api_key: str,
    top_k: int = 5,
    answer_model: str = "gpt-4o-mini",
    embedding_model_for_query: str = "text-embedding-3-large",
    persona: str = "You are a senior biotech/medical researcher focusing on microbe-based therapeutics. Answer in that voice and cite sources.",
):
    if not query.strip():
        return "", []

    coll = client.get_collection(name=collection_name)
    # embed the query
    q_emb = get_openai_embeddings([query], api_key=openai_api_key, model=embedding_model_for_query)[0]
    # Chroma similarity search
    try:
        results = coll.query(query_embeddings=[q_emb], n_results=top_k, include=['metadatas','documents','distances'])
    except Exception as e:
        raise RuntimeError(f"Chroma query failed: {e}")

    docs = []
    sources = []
    for doc, meta, dist in zip(results.get('documents', [[]])[0], results.get('metadatas', [[]])[0], results.get('distances', [[]])[0]):
        docs.append(doc)
        sources.append(meta)

    # Build context string with citations
    context = "\n\n---\n\n".join([f"[Source: {m.get('filename') or m.get('url')}]\n" + d for d, m in zip(docs, sources)])

    # Build prompt: system + user
    system_prompt = persona + "\nUse only the provided context. If answer is not present, say: 'I don't know based on the provided documents.'\nProvide a short hypothesis if appropriate, and 3 follow-up suggestions for experiments or reading. Format: Answer (markdown), Hypothesis, Follow-ups, Sources."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely in markdown."

    # Call OpenAI chat completion
    chat_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": answer_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 800,
        "temperature": 0.1,
    }
    r = requests.post(chat_url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Completion failed {r.status_code}: {r.text}")
    resp = r.json()
    answer_text = resp["choices"][0]["message"]["content"]

    # Create source list lines for citing
    citation_lines = []
    for m in sources:
        if m.get('source') == 'pdf':
            citation_lines.append(f"PDF: {m.get('filename')}")
        else:
            citation_lines.append(f"Web: {m.get('url')}")

    return answer_text, citation_lines


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(layout="wide", page_title="Advanced Biotech RAG")
st.title("Advanced RAG for Biotech & Microbe Research — PyMuPDF + ChromaDB")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts {role, content}
if "index_info" not in st.session_state:
    st.session_state.index_info = None

with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    embedding_model = st.selectbox("Embedding model", ["text-embedding-3-large", "text-embedding-3-small"], index=0)
    answer_model = st.selectbox("Answer model (chat)", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini"], index=0)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=25)
    do_ocr_images = st.checkbox("Enable OCR on images (requires pytesseract)", value=False)
    crawl_website = st.checkbox("Crawl website(s) to discover internal pages", value=False)
    persist_dir = st.text_input("ChromaDB persist dir", value="./chroma_db")
    top_k = st.slider("Retrieval top-k", min_value=1, max_value=20, value=5)

st.markdown("---")

# File upload + website input
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Upload PDFs (multiple)")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    st.subheader("Or provide websites to index")
    web_urls = st.text_area("Enter comma-separated website URLs", placeholder="https://example.com, https://another.org")

    if st.button("Index / Re-index documents"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar first.")
        else:
            with st.spinner("Indexing documents — this may take a while for 200+ PDFs..."):
                info = index_documents(
                    uploaded_files=uploaded_files,
                    web_urls=web_urls,
                    openai_api_key=openai_api_key,
                    persist_directory=persist_dir,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=embedding_model,
                    do_ocr_images=do_ocr_images and OCR_AVAILABLE,
                    crawl_website=crawl_website,
                )
                st.session_state.index_info = info

with col2:
    st.markdown("### Index status")
    if st.session_state.index_info:
        st.success(f"Indexed {st.session_state.index_info['num_chunks']} chunks — persisted to {st.session_state.index_info['persist_dir']}")
        if st.button("Clear index info"):
            st.session_state.index_info = None
    else:
        st.info("No index yet. Upload PDFs + click 'Index' to build the vector DB.")

st.markdown("---")

# Chat / Query area
st.subheader("Ask questions (RAG)")
question = st.text_input("Enter your question:")
persona_prompt = st.text_area("Persona note (how the answer should sound)", value="You are a senior biotech researcher focusing on microbe-derived therapeutics. Answer with experimental context and cautious scientific tone.")

if st.button("Get answer"):
    if not openai_api_key:
        st.error("OpenAI API key required in sidebar.")
    elif not st.session_state.index_info:
        st.error("Please index documents first.")
    else:
        try:
            st.info("Retrieving evidence and generating answer — this may take a few seconds.")
            client = init_chroma(st.session_state.index_info['persist_dir'])
            answer, cites = chroma_query_and_answer(
                client=client,
                collection_name=st.session_state.index_info['collection_name'],
                query=question,
                openai_api_key=openai_api_key,
                top_k=top_k,
                answer_model=answer_model,
                embedding_model_for_query=embedding_model,
                persona=persona_prompt,
            )
            st.markdown("### Answer")
            st.markdown(answer)

            if cites:
                st.markdown("---")
                st.markdown("**Sources:**")
                for c in cites:
                    st.markdown(f"- {c}")

            # Save to conversational memory
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Offer downloads: context + hypothesis
            if st.button("Download answer & context as JSON"):
                payload = {
                    "question": question,
                    "answer": answer,
                    "sources": cites,
                    "chat_history": st.session_state.chat_history,
                }
                b = json.dumps(payload, indent=2).encode()
                st.download_button("Download JSON", data=b, file_name="rag_answer.json")

        except Exception as e:
            st.error(f"Error during query: {e}")

# Conversation view + Follow-ups
st.markdown("---")
st.subheader("Conversation History & Suggested Follow-ups")
if st.session_state.chat_history:
    for i, h in enumerate(st.session_state.chat_history[-20:]):
        role = "You" if h["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {h['content']}")

    # quick action: suggest follow-up based on last assistant reply
    if st.button("Suggest follow-ups based on last answer"):
        last_answer = None
        for entry in reversed(st.session_state.chat_history):
            if entry['role'] == 'assistant':
                last_answer = entry['content']
                break
        if last_answer:
            # ask model to suggest follow-ups
            follow_prompt = "You are a helpful research assistant. Based on the assistant answer below, provide 5 concise follow-up questions or experiments a researcher could do next. Output as a JSON array of strings.\n\n" + last_answer
            chat_url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            payload = {"model": "gpt-4o-mini", "messages": [{"role":"user","content":follow_prompt}], "max_tokens":300, "temperature":0.2}
            r = requests.post(chat_url, headers=headers, json=payload)
            if r.status_code == 200:
                arr = r.json()["choices"][0]["message"]["content"]
                st.markdown("**Follow-up suggestions:**")
                st.markdown(arr)
            else:
                st.warning("Could not generate follow-ups.")
else:
    st.info("No conversation yet. Ask a question after indexing documents.")

# Footer
st.markdown("---")
st.caption("This app extracts text and images from PDFs using PyMuPDF, creates embeddings with OpenAI, stores vectors in ChromaDB and answers using a retrieval-augmented generation workflow. For best performance when indexing 200+ PDFs: run the Streamlit app on a machine with at least 16 GB RAM and a fast network connection to OpenAI.")

# End of file
