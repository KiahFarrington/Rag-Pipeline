"""Test suite for vector store implementations."""

import os
import sys
import tempfile
import shutil
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.vector_db.faiss_store import FAISSVectorStore
from app.vector_db.chroma_store import ChromaDBVectorStore
from app.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder


def create_sample_documents():
    """Create sample protocol documents for testing."""
    documents = [
        "Modbus RTU is a serial communication protocol developed by Modicon in 1979. It uses binary encoding for data transmission over serial networks.",
        "CAN Bus (Controller Area Network) is a robust vehicle bus standard designed to allow microcontrollers and devices to communicate.",
        "DNP3 (Distributed Network Protocol) is a set of communication protocols used between components in process automation systems.",
        "EtherCAT is an Ethernet-based fieldbus system that provides sub-microsecond precision for real-time control applications.",
        "Profibus is a standardized fieldbus communication protocol used in industrial automation systems for distributed I/O.",
        "Industrial Ethernet protocols like Ethernet/IP and Profinet provide high-speed communication for manufacturing networks."
    ]
    
    metadatas = [
        {"protocol": "modbus", "type": "overview", "year": 1979},
        {"protocol": "canbus", "type": "overview", "category": "automotive"},
        {"protocol": "dnp3", "type": "overview", "category": "automation"},
        {"protocol": "ethercat", "type": "overview", "precision": "microsecond"},
        {"protocol": "profibus", "type": "overview", "category": "fieldbus"},
        {"protocol": "ethernet", "type": "overview", "category": "industrial"}
    ]
    
    return documents, metadatas


def test_faiss_store():
    """Test FAISS vector store functionality."""
    # Create sample data
    documents, metadatas = create_sample_documents()
    
    # Create embedder
    embedder = SentenceTransformerEmbedder('all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(documents)
    
    # Initialize FAISS store
    vector_store = FAISSVectorStore()
    
    # Add documents
    doc_ids = vector_store.add_documents(documents, embeddings, metadatas)
    
    # Test basic functionality
    assert vector_store.get_document_count() == len(documents)
    assert vector_store.get_embedding_dimension() == embeddings.shape[1]
    
    # Test search
    query_text = "serial communication protocol"
    query_embedding = embedder.embed_texts([query_text])[0]  # Extract first row for 1D array
    texts, scores, metas = vector_store.search(query_embedding, top_k=3)
    
    assert len(texts) == 3
    assert len(scores) == 3
    assert len(metas) == 3
    
    # Test metadata filtering
    texts_filtered, _, _ = vector_store.search(
        query_embedding, 
        top_k=5, 
        metadata_filter={"protocol": "modbus"}
    )
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_store")
        vector_store.save(save_path)
        
        new_store = FAISSVectorStore()
        new_store.load(save_path)
        
        assert new_store.get_document_count() == len(documents)


def test_chroma_store():
    """Test ChromaDB vector store functionality."""
    # Create sample data
    documents, metadatas = create_sample_documents()
    
    # Create embedder
    embedder = SentenceTransformerEmbedder('all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(documents)
    
    # Create temporary directory for ChromaDB
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize ChromaDB store
        vector_store = ChromaDBVectorStore(
            collection_name="test_protocols",
            persist_directory=temp_dir
        )
        
        # Add documents
        doc_ids = vector_store.add_documents(documents, embeddings, metadatas)
        
        # Test basic functionality
        assert vector_store.get_document_count() == len(documents)
        assert vector_store.get_embedding_dimension() == embeddings.shape[1]
        
        # Test search
        query_text = "communication protocol"
        query_embedding = embedder.embed_texts([query_text])[0]  # Extract first row for 1D array
        texts, scores, metas = vector_store.search(query_embedding, top_k=3)
        
        assert len(texts) == 3
        assert len(scores) == 3
        assert len(metas) == 3
        
        # Test metadata filtering
        texts_filtered, _, _ = vector_store.search(
            query_embedding,
            top_k=5,
            metadata_filter={"type": "overview"}
        )
        
        # Test persistence - ChromaDB saves automatically, but call save with filepath
        vector_store.save("test_persistence")
        
        # Create new store instance to test persistence
        new_store = ChromaDBVectorStore(
            collection_name="test_protocols",
            persist_directory=temp_dir
        )
        
        assert new_store.get_document_count() == len(documents)
        
        # Explicitly close/cleanup ChromaDB resources to avoid permission errors
        del vector_store._collection
        del vector_store._client
        del new_store._collection  
        del new_store._client


def compare_stores():
    """Compare FAISS and ChromaDB performance and results."""
    documents, metadatas = create_sample_documents()
    embedder = SentenceTransformerEmbedder('all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(documents)
    
    # Test both stores
    faiss_store = FAISSVectorStore()
    faiss_store.add_documents(documents, embeddings, metadatas)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_store = ChromaDBVectorStore(
            collection_name="comparison_test",
            persist_directory=temp_dir
        )
        chroma_store.add_documents(documents, embeddings, metadatas)
        
        # Compare document counts
        faiss_count = faiss_store.get_document_count()
        chroma_count = chroma_store.get_document_count()
        
        assert faiss_count == chroma_count == len(documents)
        
        # Compare search results
        query_text = "protocol communication"
        query_embedding = embedder.embed_texts([query_text])[0]  # Extract first row for 1D array
        
        faiss_texts, faiss_scores, faiss_metas = faiss_store.search(query_embedding, top_k=3)
        chroma_texts, chroma_scores, chroma_metas = chroma_store.search(query_embedding, top_k=3)
        
        assert len(faiss_texts) == len(chroma_texts) == 3


def run_all_tests():
    """Run all vector store tests."""
    tests = [
        ("FAISS Store", test_faiss_store),
        ("ChromaDB Store", test_chroma_store),
        ("Store Comparison", compare_stores)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            # Execute test function - no return value expected
            test_func()
            print(f"{test_name}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"{test_name}: ASSERTION FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"{test_name}: CRITICAL ERROR - {e}")
            failed += 1
    
    print(f"Vector store tests: {passed}/{passed+failed} passed")
    
    if failed == 0:
        print("All vector store tests passed")
    else:
        print(f"{failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests() 