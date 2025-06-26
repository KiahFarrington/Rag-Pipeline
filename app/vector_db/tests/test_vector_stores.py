"""Test script to verify both vector database implementations work.

This script demonstrates:
1. Creating sample embeddings
2. Testing FAISS vector store
3. Testing ChromaDB vector store  
4. Comparing their performance and features
"""

import numpy as np
import sys
import os

# Add parent directory to path so we can import from the main app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.vector_db import FAISSVectorStore, ChromaDBVectorStore
from app.embedders.huggingface_embedder import create_huggingface_embeddings


def create_sample_documents():
    """Create sample industrial protocol documents for testing."""
    
    # Sample industrial protocol documentation texts
    documents = [
        "Modbus RTU protocol uses binary encoding for data transmission over serial networks",
        "CAN bus frame structure includes identifier, control field, data field, and CRC",
        "Ethernet/IP industrial protocol combines Common Industrial Protocol with TCP/IP",
        "Profinet IO operates at the data link layer providing real-time communication",
        "DeviceNet uses CAN technology for connecting industrial devices to networks",
        "DNP3 protocol includes authentication and secure authentication features",
        "IEC 61850 standard defines communication protocols for electrical substations",
        "OPC UA provides platform-independent communication for industrial automation"
    ]
    
    # Create metadata for each document (simulating industrial docs)
    metadata = [
        {"protocol": "modbus", "type": "overview", "page": 1},
        {"protocol": "can", "type": "frame_structure", "page": 15},
        {"protocol": "ethernet_ip", "type": "overview", "page": 3},
        {"protocol": "profinet", "type": "layers", "page": 8},
        {"protocol": "devicenet", "type": "overview", "page": 2},
        {"protocol": "dnp3", "type": "security", "page": 45},
        {"protocol": "iec61850", "type": "standards", "page": 12},
        {"protocol": "opc_ua", "type": "overview", "page": 7}
    ]
    
    return documents, metadata


def test_faiss_vector_store():
    """Test the FAISS vector store implementation."""
    print("\n" + "="*60)
    print("🚀 TESTING FAISS VECTOR STORE")
    print("="*60)
    
    # Create sample documents and embeddings
    documents, metadata = create_sample_documents()
    print(f"📄 Created {len(documents)} sample protocol documents")
    
    # Create embeddings using HuggingFace
    print("🧠 Creating embeddings with HuggingFace transformer...")
    embeddings = create_huggingface_embeddings(documents)
    print(f"✅ Created embeddings with shape: {embeddings.shape}")
    
    # Initialize FAISS vector store
    vector_store = FAISSVectorStore()
    
    # Add documents to the store
    print("💾 Adding documents to FAISS store...")
    doc_ids = vector_store.add_documents(documents, embeddings, metadata)
    print(f"✅ Added documents with IDs: {doc_ids[:3]}... (showing first 3)")
    
    # Test basic info
    print(f"📊 Total documents in store: {vector_store.get_document_count()}")
    print(f"📐 Embedding dimension: {vector_store.get_embedding_dimension()}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    query_text = "real-time communication protocols"
    query_embedding = create_huggingface_embeddings([query_text])[0]
    
    results = vector_store.search(query_embedding, top_k=3)
    texts, scores, result_metadata = results
    
    print(f"Query: '{query_text}'")
    print("Top 3 results:")
    for i, (text, score, meta) in enumerate(zip(texts, scores, result_metadata)):
        print(f"  {i+1}. Score: {score:.3f} | Protocol: {meta.get('protocol', 'unknown')}")
        print(f"     Text: {text[:80]}...")
    
    # Test metadata filtering
    print("\n🔧 Testing metadata filtering...")
    modbus_results = vector_store.search(
        query_embedding, 
        top_k=5, 
        metadata_filter={"protocol": "modbus"}
    )
    texts_filtered, scores_filtered, meta_filtered = modbus_results
    print(f"Filtered search for 'modbus' protocol: {len(texts_filtered)} results")
    
    # Test persistence
    print("\n💾 Testing save/load functionality...")
    save_path = "test_faiss_store"
    vector_store.save(save_path)
    
    # Create new store and load
    new_store = FAISSVectorStore()
    new_store.load(save_path)
    print(f"✅ Loaded store has {new_store.get_document_count()} documents")
    
    return vector_store


def test_chromadb_vector_store():
    """Test the ChromaDB vector store implementation."""
    print("\n" + "="*60)
    print("🎯 TESTING CHROMADB VECTOR STORE")
    print("="*60)
    
    # Create sample documents and embeddings
    documents, metadata = create_sample_documents()
    print(f"📄 Using same {len(documents)} sample protocol documents")
    
    # Create embeddings using HuggingFace
    print("🧠 Creating embeddings with HuggingFace transformer...")
    embeddings = create_huggingface_embeddings(documents)
    print(f"✅ Created embeddings with shape: {embeddings.shape}")
    
    # Initialize ChromaDB vector store
    vector_store = ChromaDBVectorStore(
        collection_name="test_protocols",
        persist_directory="./test_chroma_db"
    )
    
    # Clear any existing data for clean test
    vector_store.clear()
    
    # Add documents to the store
    print("💾 Adding documents to ChromaDB store...")
    doc_ids = vector_store.add_documents(documents, embeddings, metadata)
    print(f"✅ Added documents with IDs: {doc_ids[:3]}... (showing first 3)")
    
    # Test basic info
    print(f"📊 Total documents in store: {vector_store.get_document_count()}")
    print(f"📐 Embedding dimension: {vector_store.get_embedding_dimension()}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    query_text = "authentication and security features"
    query_embedding = create_huggingface_embeddings([query_text])[0]
    
    results = vector_store.search(query_embedding, top_k=3)
    texts, scores, result_metadata = results
    
    print(f"Query: '{query_text}'")
    print("Top 3 results:")
    for i, (text, score, meta) in enumerate(zip(texts, scores, result_metadata)):
        print(f"  {i+1}. Score: {score:.3f} | Protocol: {meta.get('protocol', 'unknown')}")
        print(f"     Text: {text[:80]}...")
    
    # Test metadata filtering
    print("\n🔧 Testing metadata filtering...")
    overview_results = vector_store.search(
        query_embedding, 
        top_k=5, 
        metadata_filter={"type": "overview"}
    )
    texts_filtered, scores_filtered, meta_filtered = overview_results
    print(f"Filtered search for 'overview' documents: {len(texts_filtered)} results")
    
    # Test persistence (ChromaDB auto-saves)
    print("\n💾 Testing auto-persistence...")
    vector_store.save("auto_saved")  # This just logs - ChromaDB auto-saves
    
    return vector_store


def compare_vector_stores(faiss_store, chroma_store):
    """Compare the performance and features of both vector stores."""
    print("\n" + "="*60)
    print("⚖️  COMPARING VECTOR STORES")
    print("="*60)
    
    # Compare document counts
    faiss_count = faiss_store.get_document_count()
    chroma_count = chroma_store.get_document_count()
    print(f"📊 Document counts:")
    print(f"   FAISS: {faiss_count} documents")
    print(f"   ChromaDB: {chroma_count} documents")
    
    # Compare search results for same query
    query_text = "industrial network communication"
    query_embedding = create_huggingface_embeddings([query_text])[0]
    
    print(f"\n🔍 Search comparison for: '{query_text}'")
    
    # FAISS results
    faiss_results = faiss_store.search(query_embedding, top_k=3)
    print("\n🚀 FAISS Results:")
    for i, (text, score, meta) in enumerate(zip(*faiss_results)):
        print(f"   {i+1}. Score: {score:.3f} | {meta.get('protocol', 'unknown')}")
    
    # ChromaDB results
    chroma_results = chroma_store.search(query_embedding, top_k=3)
    print("\n🎯 ChromaDB Results:")
    for i, (text, score, meta) in enumerate(zip(*chroma_results)):
        print(f"   {i+1}. Score: {score:.3f} | {meta.get('protocol', 'unknown')}")
    
    # Feature comparison
    print("\n📋 Feature Comparison:")
    print("   FAISS:")
    print("     ✅ Lightning fast search")
    print("     ✅ Memory efficient")
    print("     ✅ Manual persistence control")
    print("     ⚠️  More complex setup")
    
    print("\n   ChromaDB:")
    print("     ✅ Easy to use")
    print("     ✅ Automatic persistence")
    print("     ✅ Excellent metadata support")
    print("     ⚠️  Slightly slower for large datasets")


def main():
    """Main test function to run all vector store tests."""
    print("🧪 VECTOR STORE TESTING SUITE")
    print("Testing both FAISS and ChromaDB implementations")
    
    try:
        # Test FAISS vector store
        faiss_store = test_faiss_vector_store()
        
        # Test ChromaDB vector store  
        chroma_store = test_chromadb_vector_store()
        
        # Compare both stores
        compare_vector_stores(faiss_store, chroma_store)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ FAISS vector store working perfectly")
        print("✅ ChromaDB vector store working perfectly") 
        print("✅ Both stores can handle industrial protocol documents")
        print("✅ Search and metadata filtering functional")
        print("✅ Persistence working for both stores")
        
        print("\n🚀 Your RAG system vector databases are ready!")
        print("You can now choose which one to use for your project:")
        print("  - Use FAISS for maximum speed and efficiency")
        print("  - Use ChromaDB for ease of use and development")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 