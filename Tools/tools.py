from langchain_core.tools import tool
from langchain_core.messages import AIMessage
import os

# Assume the vector store will be created and imported later
vector_store = None  # This will be set by ChromaDB_Creation.py

def set_vector_store(store):
    global vector_store
    vector_store = store

@tool
def similarity_search_tool(query: str):
    """
    Similarity search tool from vector database About GeeksVisor Company.
    """
    print("---SIMILARITY SEARCH TOOL ACTIVATED---")
    if vector_store is None:
        return {"messages": [AIMessage(content="Vector store not initialized.")]}
    else:
        results = vector_store.similarity_search_with_score(query, k=2)

    if not results:
        return {"messages": [AIMessage(content="No relevant articles found.")]}
    else:
        return {"messages": [AIMessage(content=results)]}

