import streamlit as st
import os
import sys
import requests
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from utils import (
    load_index_and_metadata,
    load_embedding_model,
    embed_query,
    retrieve_top_chunks,
    build_prompt,
)

from ingest import ingest_uploaded_pdf

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"

def call_groq_api(context, query, retries=3, delay=2):
    prompt = build_prompt(context, query)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(retries):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 503:
                time.sleep(delay * (2 ** attempt))
            elif response.status_code == 404:
                return "⚠️ Model not available or access denied."
            elif response.status_code == 400 and "decommissioned" in response.text:
                return "⚠️ Model has been decommissioned."
            else:
                return f"❌ Groq API Error {response.status_code}: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"
    return "⚠️ Groq API failed after multiple retries."

@st.cache_data(show_spinner=True)
def load_resources():
    index, chunks, metadata = load_index_and_metadata()
    model = load_embedding_model()
    return index, chunks, metadata, model

st.set_page_config(page_title="🧾 Ask About Constitution", page_icon="📜")
st.title("📜 Ask About Constitution")
st.subheader("India • China • USA • Japan • Germany")

# File upload
uploaded_file = st.file_uploader("📤 Upload a Constitution PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing and indexing PDF..."):
        num_chunks = ingest_uploaded_pdf(uploaded_file, filename=uploaded_file.name)
        st.success(f"{uploaded_file.name} processed with {num_chunks} chunks.")

# Load RAG resources
index, chunks, metadata, embed_model = load_resources()

# Initialize memory
if "history" not in st.session_state:
    st.session_state.history = []

# Chat interface
with st.container():
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Lex:** {a}")

query = st.chat_input("Type your constitutional question...", key="chat_input")

if query:
    if len(query) < 3:
        st.warning("Please enter a more complete question.")
    else:
        with st.spinner("Thinking..."):
            query_embedding = embed_query(query, embed_model)
            top_chunks = retrieve_top_chunks(query_embedding, index, chunks, metadata)
            context = "\n\n".join(top_chunks)
            answer = call_groq_api(context, query)

        st.session_state.history.append((query, answer))
        st.rerun()
