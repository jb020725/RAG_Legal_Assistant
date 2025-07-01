import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load FAISS index and metadata ===
def load_index_and_metadata(index_folder="index"):
    index = faiss.read_index(os.path.join(index_folder, "rag_index.faiss"))
    with open(os.path.join(index_folder, "rag_index.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]

# === Load embedding model ===
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# === Embed the user query ===
def embed_query(query, model):
    return model.encode([query])[0].astype("float32")

# === Retrieve top-k relevant chunks ===
def retrieve_top_chunks(query_embedding, index, chunks, metadata, top_k=5):
    scores, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0]]

# === System prompt ===
SYSTEM_PROMPT = """You are Lex, a friendly AI assistant. 
Respond clearly and conversationally, using the provided legal context if relevant.
Be natural and curious. Avoid phrases like 'According to the context' or 'Based on the above.'
"""

# === Build the final prompt for the LLM ===
def build_prompt(context, query):
    return f"""
### System:
{SYSTEM_PROMPT}

### Legal Context:
{context}

### User Question:
{query}

### Lex's Answer:
"""
