"""Complete RAG system demonstration with Augmentation.

This demonstrates the full RAG pipeline:
Data Loading → Chunking → Embedding → Vector Storage → Retrieval → **AUGMENTATION**

NEW: Added text generation using free language models!
"""

import os
from chunkers.semantic_chunker import chunk_by_semantics
from embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding
from vector_db.faiss_store import FAISSVectorStore

# Import both retriever options - MODULAR!
from retriever.dense_retriever import DenseRetriever
from retriever.hybrid_retriever import HybridRetriever

# NEW: Import free augmentation generators
from augmented_generation.ollama_generator import create_ollama_generator, list_available_models as list_ollama_models
from augmented_generation.huggingface_generator import create_huggingface_generator, list_available_models as list_hf_models


def load_sample_data(file_path: str) -> str:
    """Load sample text data from file."""
    # Check if file exists, handle error early
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample data file not found: {file_path}")
    
    # Read and return file contents
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def demonstrate_augmentation_options():
    """Show available free augmentation options."""
    print("\n=== FREE AUGMENTATION OPTIONS ===")
    
    print("\n1. OLLAMA (100% Free, Local):")
    print("   - Runs entirely on your machine")
    print("   - No API keys required")
    print("   - No internet needed after setup")
    ollama_models = list_ollama_models()
    for model, desc in ollama_models.items():
        print(f"   • {model}: {desc}")
    
    print("\n2. HUGGING FACE TRANSFORMERS (100% Free, Local):")
    print("   - Download once, use forever")
    print("   - No API costs")
    print("   - Good for resource-constrained environments")
    hf_models = list_hf_models()
    for model, desc in list(hf_models.items())[:4]:  # Show first 4
        print(f"   • {model}: {desc}")
    
    print("\n3. Setup Instructions:")
    print("   Ollama: Visit https://ollama.ai/ → Download → Run 'ollama pull llama2'")
    print("   HuggingFace: pip install transformers torch (auto-downloads models)")


def run_complete_rag_with_augmentation():
    """Run the complete RAG system including text generation."""
    print("=== COMPLETE RAG SYSTEM WITH AUGMENTATION ===")
    
    # Step 1: Load document data
    document_text = load_sample_data("data/sample.txt")
    print(f"✓ Loaded document: {len(document_text)} characters")
    
    # Step 2: Chunk the document
    chunks = chunk_by_semantics(document_text)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings for chunks
    chunk_embeddings = create_tfidf_embeddings(chunks)
    print(f"✓ Generated embeddings: {chunk_embeddings.shape}")
    
    # Step 4: Store in vector database
    vector_store = FAISSVectorStore()
    vector_store.add_documents(chunks, chunk_embeddings)
    print(f"✓ Stored {len(chunks)} vectors in FAISS")
    
    # Step 5: Prepare query
    query = "What is the main communication protocol mentioned?"
    query_embedding = create_single_tfidf_embedding(query)
    print(f"\n📝 Query: '{query}'")
    
    # Step 6: Retrieve relevant chunks
    retriever = HybridRetriever(vector_weight=0.6, keyword_weight=0.4)
    retrieved_chunks = retriever.retrieve(vector_store, query_embedding, query, top_k=3)
    
    print(f"\n📚 Retrieved {len(retrieved_chunks)} relevant chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"   {i+1}. {chunk[:100]}...")
    
    # Step 7: AUGMENTATION - Generate response using LLM
    print(f"\n🤖 AUGMENTATION - Generating response...")
    
    # Try different generators in order of preference
    generators_to_try = [
        ("Ollama", lambda: create_ollama_generator("llama2")),
        ("Hugging Face", lambda: create_huggingface_generator("google/flan-t5-base", use_small_model=True))
    ]
    
    response_generated = False
    
    # Try each generator until one works
    for generator_name, generator_factory in generators_to_try:
        try:
            print(f"   Trying {generator_name} generator...")
            generator = generator_factory()
            
            # Generate response using retrieved chunks as context
            result = generator.generate_response(query, retrieved_chunks)
            
            # Check if generation was successful
            if result['success']:
                print(f"\n✅ SUCCESS with {generator_name}!")
                print(f"Model used: {result.get('model_used', 'Unknown')}")
                print(f"Context length: {result.get('context_length', 0)} characters")
                print(f"\n💬 GENERATED RESPONSE:")
                print(f"   {result['response']}")
                response_generated = True
                break
            else:
                print(f"   ❌ {generator_name} failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ {generator_name} error: {str(e)}")
    
    # If no generators worked, show fallback message
    if not response_generated:
        print(f"\n⚠️  No generators available. Install one of these options:")
        print(f"   • Ollama: https://ollama.ai/ (Recommended - completely free)")
        print(f"   • Hugging Face: pip install transformers torch")
        print(f"\n📝 Retrieved Context (without generation):")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"   Source {i+1}: {chunk}")
    
    print(f"\n=== COMPLETE RAG PIPELINE FINISHED ===")


def main():
    """Run the RAG system with modular components."""
    # Show available augmentation options
    demonstrate_augmentation_options()
    
    print(f"\n" + "="*60)
    
    # Run complete RAG pipeline with augmentation
    run_complete_rag_with_augmentation()
    
    print(f"\n🎉 RAG SYSTEM MODULARITY:")
    print(f"   ✓ Chunkers: Semantic, Fixed-length")
    print(f"   ✓ Embedders: TF-IDF, Sentence Transformers, HuggingFace")
    print(f"   ✓ Vector Stores: FAISS, Chroma")
    print(f"   ✓ Retrievers: Dense, Hybrid, Similarity")
    print(f"   ✓ Generators: Ollama, HuggingFace, (OpenAI when needed)")


if __name__ == "__main__":
    main() 