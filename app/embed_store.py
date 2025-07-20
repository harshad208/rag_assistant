import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import logging

DATA_PATH = "./data"
DB_PATH = "./db/chroma" 
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def create_vector_store():
    try:
        """
        Loads documents, splits them into chunks, creates embeddings,
        and stores them in a ChromaDB vector store.
        """
        logging.info("Starting document ingestion and vector store creation...")

        logging.info(f"Loading documents from: {DATA_PATH}")
        loader = DirectoryLoader(DATA_PATH, glob="**/*", show_progress=True)
        documents = loader.load()
        if not documents:
            logging.info("No documents found. Please add your documents to the 'data' directory.")
            return

        logging.info(f"Loaded {len(documents)} document(s).")

        logging.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(texts)} chunks.")

        logging.info(f"Creating embeddings using model: {EMBEDDING_MODEL}")

        model_kwargs = {'device': 'cpu'}

        encode_kwargs = {'normalize_embeddings': True}
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        logging.info("Creating and persisting vector store...")
        
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
            
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )

        logging.info("Vector store created successfully!")
        logging.info(f"Number of vectors in store: {vector_store._collection.count()}")
    except Exception as e:
        logging.error(f"Error While creating error: {str(e)}", exc_info=True)
    

if __name__ == '__main__':
    create_vector_store()