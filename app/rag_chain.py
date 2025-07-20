from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
import os
import logging

DB_PATH = "./db/chroma"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = "phi3"

def get_rag_chain(selected_files: List[str] = None):
    """
    Creates and returns a RAG chain, optionally filtered by selected files.

    Args:
        selected_files (List[str], optional): A list of source file paths to filter the search.
                                              If None or empty, searches all documents.
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    search_kwargs = {'k': 3}
    if selected_files:
        filter_paths = [os.path.join("data", f) for f in selected_files]
        search_kwargs['filter'] = {"source": {"$in": filter_paths}}
        logging.debug(f"DEBUG: Retriever is filtering for sources: {filter_paths}")

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434")

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

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain