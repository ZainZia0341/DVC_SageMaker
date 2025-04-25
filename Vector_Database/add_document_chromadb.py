import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from LLMs.llm import load_embedding_model  # embedding model will be fetched from llm.py
from Tools import set_vector_store

def create_vector_store_from_pdfs(pdf_file_paths: list, collection_name: str = None, persist_directory: str = "./chroma_langchain_db"):
    """
    Loads the provided PDF files, splits them into chunks, creates a new ChromaDB collection,
    adds the documents to the vector store, and sets the vector store for later use.
    """
    if collection_name is None:
        collection_name = f"collection_{uuid.uuid4().hex}"
    
    docs = []
    for pdf_path in pdf_file_paths:
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            continue
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {pdf_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        print(f"Number of chunks from {pdf_path}: {len(chunks)}")
        docs.extend(chunks)
    
    # Fetch embedding model from llm.py
    embedding_model = load_embedding_model()
    os.environ["GOOGLE_CLOUD_DISABLE_DETECTION"] = "True"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    print("Vector store created with collection name:", collection_name)
    vector_store.add_documents(docs)
    print("Documents added to vector store!")
    
    # Set the vector store for use in Tools.py
    set_vector_store(vector_store)
    
    return vector_store
