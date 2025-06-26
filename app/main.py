"""Simple RAG system demonstration.

This demonstrates a basic RAG pipeline:
Data Loading → Chunking → Embedding → Vector Storage → Retrieval → Generation
"""

import os
from chunkers.semantic_chunker import chunk_by_semantics
from embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding
from vector_db.faiss_store import FAISSVectorStore

# Import both retriever options - MODULAR!
from retriever.dense_retriever import DenseRetriever
from retriever.hybrid_retriever import HybridRetriever


def load_sample_data(file_path: str) -> str:
    """Load sample text data from file."""
    # Check if file exists, handle error early
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample data file not found: {file_path}")
    
    # Read and return file contents
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    """Run the basic RAG system with modular retriever choice."""
    print("=== BASIC RAG SYSTEM ===")
    
    # Step 1: Load document data
    document_text = load_sample_data("data/sample.txt")
    print(f"Loaded document: {len(document_text)} characters")
    
    # Step 2: Chunk the document
    chunks = chunk_by_semantics(document_text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings for chunks
    chunk_embeddings = create_tfidf_embeddings(chunks)
    print(f"Generated embeddings: {chunk_embeddings.shape}")
    
    # Step 4: Store in vector database
    vector_store = FAISSVectorStore()
    vector_store.add_documents(chunks, chunk_embeddings)
    print(f"Stored {len(chunks)} vectors in FAISS")
    
    # Step 5: Prepare query
    query = "communication protocol"
    query_embedding = create_single_tfidf_embedding(query)
    
    # Step 6: MODULAR RETRIEVAL - Choose your strategy!
    print(f"\n=== RETRIEVAL COMPARISON ===")
    
    # Option 1: Dense retriever (pure vector similarity)
    dense_retriever = DenseRetriever(similarity_threshold=0.1)
    dense_results = dense_retriever.retrieve(vector_store, query_embedding, top_k=3)
    
    print(f"\nDENSE RETRIEVER:")
    print(f"Query: '{query}'")
    print(f"Retrieved {len(dense_results)} chunks:")
    for i, chunk in enumerate(dense_results):
        print(f"  {i+1}. {chunk[:80]}...")
    
    # Option 2: Hybrid retriever (vector + keyword matching)
    hybrid_retriever = HybridRetriever(vector_weight=0.6, keyword_weight=0.4)
    hybrid_results = hybrid_retriever.retrieve(vector_store, query_embedding, query, top_k=3)
    
    print(f"\nHYBRID RETRIEVER:")
    print(f"Query: '{query}'")
    print(f"Retrieved {len(hybrid_results)} chunks:")
    for i, chunk in enumerate(hybrid_results):
        print(f"  {i+1}. {chunk[:80]}...")
    
    print("\n=== RAG SYSTEM COMPLETE ===")
    print("MODULARITY: Easily swap retrievers by changing 1 line!")


if __name__ == "__main__":
    main() 