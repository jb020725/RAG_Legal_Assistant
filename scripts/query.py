import os
import requests
import time
from dotenv import load_dotenv
from utils import (
    load_index_and_metadata,
    load_embedding_model,
    embed_query,
    retrieve_top_chunks,
    build_prompt,
)

load_dotenv()

# === Config ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"
MAX_RETRIES = 3
RETRY_DELAY = 2

def call_groq_api(context, query, retries=MAX_RETRIES, delay=RETRY_DELAY):
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
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
                continue
            elif response.status_code == 404:
                return "⚠️ Error: This model may have been removed or access is denied."
            elif response.status_code == 400 and "decommissioned" in response.text:
                return "⚠️ This model has been decommissioned by Groq. Please update your configuration."
            else:
                return f"❌ Groq API Error {response.status_code}: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"
    return "⚠️ Groq API unavailable after multiple attempts. Please try again later."

def main():
    print("🔍 Loading FAISS index and metadata...")
    index, chunks, metadata = load_index_and_metadata()

    print("🧠 Loading embedding model...")
    model = load_embedding_model()

    print("📜 Constitution QA CLI")
    print("Type your question (or 'exit' to quit):\n")
    
    while True:
        query = input("🔍 Your Question: ").strip()
        if query.lower() == "exit":
            break
        if len(query) < 3:
            print("⚠️ Please enter a more complete question.\n")
            continue

        query_embedding = embed_query(query, model)
        top_chunks = retrieve_top_chunks(query_embedding, index, chunks, metadata)
        context = "\n\n".join(top_chunks)

        answer = call_groq_api(context, query)
        print(f"\n🤖 Lex's Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
