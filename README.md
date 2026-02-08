# 🤖 RAG-Based Chatbot using Ollama (Local LLM)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions from their **own documents (PDF/TXT/MD)**.  
It uses **Ollama** to run a **local Large Language Model (LLM)** and **ChromaDB** for vector-based semantic search.

👉 No OpenAI API key required  
👉 Fully local & privacy-friendly  

---

## ✨ Features

- 📄 Upload and chat with **PDF / TXT / Markdown** files  
- 🧠 Semantic search using **HuggingFace embeddings**  
- 🗂️ Persistent **vector database (ChromaDB)**  
- 🤖 Local LLM inference using **Ollama**  
- 🧾 Source references for answers  
- 🖥️ Interactive **Streamlit chat UI**

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
1. **Information Retrieval** – fetch relevant document chunks
2. **Text Generation** – generate answers using an LLM

This reduces hallucinations and ensures answers are **grounded in real documents**.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – frontend UI
- **LangChain**
- **Ollama** – local LLM (LLaMA / Mistral)
- **HuggingFace Embeddings**
- **ChromaDB** – vector store
- **PyPDF** – PDF loading

---

## 📁 Project Structure
