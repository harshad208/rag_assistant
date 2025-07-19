# In app/embed_store.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DATA_PATH = "./data"
DB_PATH = "./db/chroma" # Be more specific with the path
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def create_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a ChromaDB vector store.
    """
    print("Starting document ingestion and vector store creation...")

    # 1. Load documents
    print(f"Loading documents from: {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="**/*", show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Please add your documents to the 'data' directory.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # 2. Split documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. Create embeddings (with the fix)
    print(f"Creating embeddings using model: {EMBEDDING_MODEL}")
    # --- FIX APPLIED HERE ---
    # Explicitly specify the device to load the model onto.
    model_kwargs = {'device': 'cpu'}
    # Specify whether to normalize embeddings. This is a best practice.
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # --- END OF FIX ---

    # 4. Create and persist the ChromaDB vector store
    print("Creating and persisting vector store...")
    # Ensure the directory exists
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Vector store created successfully!")
    print(f"Number of vectors in store: {vector_store._collection.count()}")

if __name__ == '__main__':
    create_vector_store()