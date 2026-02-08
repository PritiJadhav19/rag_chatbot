import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag import load_documents, split_documents, build_or_update_vectorstore, answer_question

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG-Based Chatbot (PDF/TXT)")

with st.sidebar:
    st.header("📄 Upload Documents")
    files = st.file_uploader("Upload PDFs/TXT", type=["pdf", "txt", "md"], accept_multiple_files=True)

    st.markdown("---")
    st.header("⚙️ Settings")
    k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=4, step=1)
    chunk_size = st.slider("Chunk Size", min_value=300, max_value=1500, value=900, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150, step=25)

    st.markdown("---")
    st.caption("Tip: For best answers, set OPENAI_API_KEY in your environment.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False


def index_files(uploaded_files):
    # Save uploads to temp files
    temp_paths = []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            temp_paths.append(tmp.name)

    docs = load_documents(temp_paths)
    if not docs:
        return False, "No supported documents found."

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_or_update_vectorstore(chunks)

    # cleanup temp files
    for p in temp_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    return True, f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)."


# Indexing button
if files and not st.session_state.indexed:
    if st.button("📌 Build / Update Knowledge Base"):
        ok, msg = index_files(files)
        st.session_state.indexed = ok
        st.success(msg) if ok else st.error(msg)

if not st.session_state.indexed:
    st.info("Upload documents and click **Build / Update Knowledge Base** to start chatting.")

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.write("\n".join([f"- {s}" for s in m["sources"]]))

# Chat input
prompt = st.chat_input("Ask something from your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not os.path.exists("chroma_db") or not os.listdir("chroma_db"):
            st.error("Vector DB not found. Upload docs and build the knowledge base first.")
        else:
            ans, sources = answer_question(prompt, k=k)
            st.markdown(ans)
            with st.expander("Sources"):
                st.write("\n".join([f"- {s}" for s in sources]))

            st.session_state.messages.append(
                {"role": "assistant", "content": ans, "sources": sources}
            )