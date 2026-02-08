import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_ollama import ChatOllama  # <-- Ollama LLM (local)

DB_DIR = "chroma_db"


def load_documents(file_paths: List[str]):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path.lower())[1]
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext in [".txt", ".md"]:
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def split_documents(docs, chunk_size=900, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_update_vectorstore(docs) -> Chroma:
    embeddings = get_embeddings()

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vectordb.add_documents(docs)
    else:
        vectordb = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=DB_DIR,
        )

    vectordb.persist()
    return vectordb


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def retrieve_context(vectordb: Chroma, query: str, k: int = 4):
    return vectordb.similarity_search(query, k=k)


def format_sources(docs) -> List[str]:
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        if page is not None:
            sources.append(f"{os.path.basename(src)} (page {page+1})")
        else:
            sources.append(f"{os.path.basename(src)}")

    seen = set()
    uniq = []
    for s in sources:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def answer_question(query: str, k: int = 4) -> Tuple[str, List[str]]:
    vectordb = load_vectorstore()
    retrieved = retrieve_context(vectordb, query, k=k)

    context = "\n\n".join([f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(retrieved)])
    sources = format_sources(retrieved)

    # Ollama local model (change model name if you pulled a different one)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        temperature=0.2,
    )

    prompt = f"""You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the uploaded documents."

Context:
{context}

Question:
{query}

Answer:"""

    resp = llm.invoke(prompt)
    return resp.content.strip(), sources