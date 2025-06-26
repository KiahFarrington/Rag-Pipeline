"""Vector database package for storing and searching embeddings.

This package contains different vector storage and similarity search
implementations for the RAG system. All implementations follow the
BaseVectorStore interface for easy switching between databases.

Available vector stores:
- FAISSVectorStore: High-performance vector store using FAISS for similarity search
- ChromaDBVectorStore: User-friendly vector store with automatic persistence
- BaseVectorStore: Abstract base class defining the common interface

Example usage:
    # Using FAISS (faster, more memory efficient)
    from app.vector_db import FAISSVectorStore
    vector_store = FAISSVectorStore(embedding_dim=384)
    
    # Using ChromaDB (easier to use, automatic persistence)
    from app.vector_db import ChromaDBVectorStore
    vector_store = ChromaDBVectorStore(collection_name="protocols")
    
    # Both support the same interface
    doc_ids = vector_store.add_documents(texts, embeddings, metadata)
    results = vector_store.search(query_embedding, top_k=5)

Testing:
    # Run comprehensive tests for both vector stores
    python app/vector_db/tests/test_vector_stores.py
"""

# Import all vector store implementations
from .base_vector_store import BaseVectorStore
from .faiss_store import FAISSVectorStore
from .chroma_store import ChromaDBVectorStore

# Define what gets imported with "from app.vector_db import *"
__all__ = [
    'BaseVectorStore',
    'FAISSVectorStore', 
    'ChromaDBVectorStore'
] 