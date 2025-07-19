# In app/rag_chain.py

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
import os

# --- Configuration ---
DB_PATH = "./db/chroma"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = "phi3"

# --- MODIFICATION: The function now accepts a list of selected files ---
def get_rag_chain(selected_files: List[str] = None):
    """
    Creates and returns a RAG chain, optionally filtered by selected files.

    Args:
        selected_files (List[str], optional): A list of source file paths to filter the search.
                                              If None or empty, searches all documents.
    """
    # 1. Initialize Embeddings
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 2. Load the Vector Store
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # --- MODIFICATION: Create a filter for the retriever ---
    search_kwargs = {'k': 3}
    if selected_files:
        # Construct paths that are OS-agnostic and match DirectoryLoader's output
        filter_paths = [os.path.join("data", f) for f in selected_files]
        search_kwargs['filter'] = {"source": {"$in": filter_paths}}
        print(f"DEBUG: Retriever is filtering for sources: {filter_paths}") # For debugging

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # 3. Initialize the LLM
    llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434")

    # 4. Create the RAG Prompt Template
    template = """
    You are an assistant for question-answering tasks.
    Use only the following pieces of retrieved context to answer the question.
    If you don't know the answer from the provided context, just say that you don't know.
    Do not use any other information. Keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 5. Create the RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain