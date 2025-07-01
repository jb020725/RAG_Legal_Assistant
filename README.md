
# 🧑‍⚖️ RAG Legal Assistant

This is a **Retrieval-Augmented Generation (RAG)** powered legal assistant that answers user queries by retrieving and summarizing relevant information from legal documents (like constitutions or acts). It's designed to help users search, explore, and understand complex legal content in natural language.

---

## 🚀 Live App

👉 [Use the Legal Assistant](https://legal-assiatant-by-jb.streamlit.app/)

---

## 📂 Project Structure

```
RAG_Legal_Assistant/
│
├── app/
│   ├── main.py              # Streamlit frontend
│   └── helper.py            # Backend logic for retrieval and generation
│
├── data/
│   └── *.pdf                # Input legal documents
│
├── index/
│   ├── rag_index.faiss      # FAISS vector index
│   └── rag_index.pkl        # Metadata (mapping of chunks to source)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Tech Stack

- **LLM**: [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: FAISS
- **Frontend**: Streamlit
- **RAG Framework**: Manual RAG pipeline using `langchain`-like custom logic

---

## ⚙️ Features

- 📄 Upload and parse legal PDFs
- 🔍 Retrieve relevant chunks based on semantic similarity
- 🧠 Query answered using powerful open-weight LLM via API
- 🧾 Source documents shown for transparency
- 💡 Clean, simple UI using Streamlit

---

## 🔧 Installation (for local use)


git clone https://github.com/jb020725/RAG_Legal_Assistant.git
cd RAG_Legal_Assistant

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate  # Or use venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/main.py
```

---

## 📌 Future Work

- ✅ Add citation-level document traceability  
- ✅ Use Mixtral or other LLMs via Groq or OpenRouter  
- 🔒 Add document-level access control  
- 🧾 Legal summarizer for full acts  
- 📤 File upload from UI (WIP)

---

## 📫 Contact

**Made by Janak Bhat**  
📧 janakbhat34@gmail.com  
🔗 [GitHub Repo](https://github.com/jb020725/RAG_Legal_Assistant)
