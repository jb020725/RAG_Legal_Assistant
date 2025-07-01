import os
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdf_file(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ingest_uploaded_pdf(file, filename="uploaded.pdf", index_folder="index"):
    text = extract_text_from_pdf_file(file)
    chunks = chunk_text(text)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)
    embeddings = np.array(embeddings).astype("float32")

    index = build_faiss_index(embeddings)

    os.makedirs(index_folder, exist_ok=True)
    faiss.write_index(index, os.path.join(index_folder, "rag_index.faiss"))
    with open(os.path.join(index_folder, "rag_index.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": [{"source": filename} for _ in chunks]}, f)

    return len(chunks)
