import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from LLMs.llm import load_embedding_model
from Tools.tools import set_vector_store

embedding_model = load_embedding_model()

def create_vector_store(pdf_path: str, persist_directory: str = "./chroma_langchain_db", collection_name: str = "example_collection"):
    if not os.path.exists(pdf_path):
        print("No pdf file is found to load")
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {pdf_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        print(f"Number of chunks: {len(docs)}")

        # Use embedding model from llm.py

        # Ensure that GCE detection is disabled if necessary
        # os.environ["GOOGLE_CLOUD_DISABLE_DETECTION"] = "True"

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        print("Vector store created. Adding documents...")

        vector_store.add_documents(docs)
        print("Documents added to vector store!")
        
        # Set vector store in Tools module for later use
        set_vector_store(vector_store)
    
    return vector_store
