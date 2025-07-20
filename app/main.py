import streamlit as st
import os
import time
import chromadb
from rag_chain import get_rag_chain
from database_utils import init_database, log_query
from embed_store import create_vector_store

st.set_page_config(page_title="Intelligent RAG Assistant", layout="wide")

DATA_DIR = "data"
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'chroma')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
init_database()

def get_processed_documents_from_db():
    if not os.path.exists(DB_PATH):
        return set()
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name="langchain")
        metadata = collection.get(include=["metadatas"])
        if not metadata['ids']:
            return set()
        processed_sources = {meta['source'] for meta in metadata['metadatas']}
        return processed_sources
    except Exception:
        return set()

def get_unprocessed_files():
    disk_files = {os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)}
    processed_files = get_processed_documents_from_db()
    unprocessed_paths = disk_files - processed_files
    return [os.path.basename(f) for f in unprocessed_paths]


with st.sidebar:
    st.header("🛠️ Document Management")

    uploaded_files = st.file_uploader(
        "Upload new documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        files_saved = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_saved += 1
        
        if files_saved > 0:
            st.success(f"{files_saved} new document(s) uploaded. The 'Process' button is now available below.")

    st.divider()

    unprocessed_files = get_unprocessed_files()
    if unprocessed_files:
        st.info(f"Found {len(unprocessed_files)} unprocessed document(s):")
        for f in unprocessed_files:
            st.write(f"- _{f}_")
        
        if st.button("Process New Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                create_vector_store()

                st.cache_resource.clear()
                st.success("Knowledge base updated successfully!")
                time.sleep(2)
                st.rerun() 

    st.header("📄 Select Documents to Query")
    
    processed_docs_basenames = [os.path.basename(f) for f in get_processed_documents_from_db()]

    if not processed_docs_basenames:
        st.warning("No processed documents available. Please upload and process files.")
        selected_docs = []
    else:
        selected_docs = st.multiselect(
            "Choose documents to use as context:",
            options=sorted(processed_docs_basenames),
            default=sorted(processed_docs_basenames)
        )


st.title("🤖 Intelligent RAG AI Assistant")
st.markdown("Upload documents and process them. Then, ask questions based on your selection.")

if not selected_docs:
    st.warning("Please select at least one processed document from the sidebar to begin.")
else:
    @st.cache_resource
    def load_rag_chain(selected_files):
        return get_rag_chain(selected_files=selected_files)

    rag_chain = load_rag_chain(tuple(sorted(selected_docs))) 

    st.header("💬 Ask Your Questions")
    question = st.text_input(
        "Ask a question about the selected document(s):",
        key="chat_input"
    )

    if question:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(question)
            st.success("Answer:")
            st.write(response)
            log_query(question, response)