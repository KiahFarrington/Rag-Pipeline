"""Complete RAG system demonstration with Augmentation.

This demonstrates the full RAG pipeline:
Data Loading → Chunking → Embedding → Vector Storage → Retrieval → **AUGMENTATION**

NEW: Added text generation using free language models!
"""

import os
from typing import List, Optional
from chunkers.semantic_chunker import chunk_by_semantics
from chunkers.fixed_length_chunker import chunk_by_fixed_length
from embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding
from embedders.sentence_transformer_embedder import create_sentence_transformer_embeddings, create_single_sentence_transformer_embedding
from vector_db.faiss_store import FAISSVectorStore

# Import both retriever options - MODULAR!
from retriever.dense_retriever import DenseRetriever
from retriever.hybrid_retriever import HybridRetriever

# NEW: Import free augmentation generators
from augmented_generation.ollama_generator import create_ollama_generator, list_available_models as list_ollama_models
from augmented_generation.huggingface_generator import create_huggingface_generator, list_available_models as list_hf_models


def load_document(file_path: Optional[str] = None) -> str:
    """Load document text from file with interactive prompt if needed."""
    while True:
        # Use provided path or ask user
        if not file_path:
            file_path = input("\nEnter path to your document (or 'sample' for sample.txt): ").strip()
            
        # Handle sample data case
        if file_path.lower() == 'sample':
            file_path = "app/data/sample.txt"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please try again.")
            file_path = None  # Reset to ask again
        except Exception as e:
            print(f"Error reading file: {e}")
            file_path = None  # Reset to ask again


def get_chunking_method() -> str:
    """Let user choose chunking method."""
    while True:
        print("\nChunking Methods:")
        print("1. Semantic Chunking (chunks by meaning)")
        print("2. Fixed Length Chunking (chunks by size)")
        choice = input("Choose chunking method (1-2): ").strip()
        
        if choice in ['1', '2']:
            return 'semantic' if choice == '1' else 'fixed'
        print("Invalid choice. Please enter 1 or 2.")


def get_embedding_method() -> str:
    """Let user choose embedding method."""
    while True:
        print("\nEmbedding Methods:")
        print("1. TF-IDF (faster, simpler)")
        print("2. Sentence Transformers (better semantic understanding)")
        choice = input("Choose embedding method (1-2): ").strip()
        
        if choice in ['1', '2']:
            return 'tfidf' if choice == '1' else 'transformer'
        print("Invalid choice. Please enter 1 or 2.")


def create_chunks(text: str, method: str) -> List[str]:
    """Create chunks using selected method."""
    if method == 'semantic':
        return chunk_by_semantics(text)
    return chunk_by_fixed_length(text)


def create_embeddings(chunks: List[str], method: str):
    """Create embeddings using selected method."""
    if method == 'tfidf':
        return create_tfidf_embeddings(chunks)
    return create_sentence_transformer_embeddings(chunks)


def create_query_embedding(query: str, method: str):
    """Create query embedding using selected method."""
    if method == 'tfidf':
        return create_single_tfidf_embedding(query)
    return create_single_sentence_transformer_embedding(query)


def display_results(query: str, chunks_with_scores: List[tuple], generator_result: Optional[dict] = None):
    """Display search results and generated response in a formatted way."""
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)
    
    if generator_result and generator_result.get('success'):
        print("\nGenerated Response:")
        print("-"*40)
        print(generator_result['response'])
        print(f"\nModel Used: {generator_result.get('model_used', 'Unknown')}")
        print("-"*40)
    
    print("\nRetrieved Chunks (with similarity scores):")
    print("-"*80)
    for i, (chunk, score) in enumerate(chunks_with_scores, 1):
        # Truncate long chunks for display
        display_chunk = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"\n{i}. Similarity Score: {score:.4f}")
        print(f"   {display_chunk}")
    print("="*80)


def run_interactive_rag():
    """Run the RAG system interactively."""
    print("Welcome to the Interactive RAG System!")
    
    # Step 1: Load document
    document_text = load_document()
    
    # Step 2: Choose chunking method
    chunking_method = get_chunking_method()
    chunks = create_chunks(document_text, chunking_method)
    print(f"\nCreated {len(chunks)} chunks.")
    
    # Step 3: Choose embedding method
    embedding_method = get_embedding_method()
    chunk_embeddings = create_embeddings(chunks, embedding_method)
    
    # Step 4: Store in vector database
    vector_store = FAISSVectorStore()
    vector_store.add_documents(chunks, chunk_embeddings)
    
    # Step 5: Interactive query loop
    while True:
        # Get query from user
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        # Create query embedding
        query_embedding = create_query_embedding(query, embedding_method)
        
        # Retrieve chunks with scores
        retriever = DenseRetriever()
        chunks_with_scores = retriever.retrieve_with_scores(vector_store, query_embedding, top_k=5)
        
        # Try to generate response
        generator_result = None
        try:
            generator = create_ollama_generator("llama2")
            chunks_text = [chunk for chunk, _ in chunks_with_scores]
            generator_result = generator.generate_response(query, chunks_text)
        except Exception:
            try:
                generator = create_huggingface_generator("google/flan-t5-base", use_small_model=True)
                chunks_text = [chunk for chunk, _ in chunks_with_scores]
                generator_result = generator.generate_response(query, chunks_text)
            except Exception:
                print("\nNote: AI generation unavailable. Showing retrieved chunks only.")
        
        # Display results
        display_results(query, chunks_with_scores, generator_result)


def main():
    """Run the interactive RAG system."""
    try:
        run_interactive_rag()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThank you for using the RAG system!")


if __name__ == "__main__":
    main() 