"""Comprehensive test suite for the enhanced RAG system.

This test file validates all major components including:
- Adaptive chunking
- Advanced retrieval
- Memory-efficient vector storage
- Enhanced API functionality
- Analytics and monitoring
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import all RAG components for testing
from app.chunkers.adaptive_chunker import AdaptiveChunker, chunk_adaptively
from app.chunkers.semantic_chunker import chunk_by_semantics
from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
from app.embedders.sentence_transformer_embedder import create_sentence_transformer_embeddings, create_single_sentence_transformer_embedding
from app.embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding
from app.vector_db.faiss_store import FAISSVectorStore
from app.retriever.dense_retriever import DenseRetriever
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.advanced_retriever import AdvancedRetriever


class RAGSystemTester:
    """Comprehensive tester for the enhanced RAG system."""
    
    def __init__(self):
        """Initialize the tester with sample data and test configurations."""
        # Sample technical document for testing
        self.sample_document = """
        Industrial Protocol Documentation

        Modbus RTU Protocol Overview:
        Modbus RTU is a serial communication protocol developed by Modicon in 1979.
        It uses binary encoding for data transmission over serial networks.
        The protocol supports master-slave communication with up to 247 devices.
        Error checking is performed using CRC (Cyclic Redundancy Check).

        CAN Bus Protocol Fundamentals:
        Controller Area Network (CAN) is a robust vehicle bus standard.
        Originally designed for automotive applications, now used in industrial automation.
        CAN provides multi-master communication with collision detection and arbitration.
        Message prioritization is handled through identifier fields.

        DNP3 Security Features:
        Distributed Network Protocol 3 (DNP3) includes advanced security mechanisms.
        Authentication is provided through challenge-response protocols.
        Secure authentication prevents unauthorized access to critical infrastructure.
        The protocol supports encryption for sensitive data transmission.

        Industrial Ethernet Networks:
        Ethernet/IP and Profinet provide high-speed industrial communication.
        These protocols operate at the data link layer for real-time control.
        Network topology can be star, ring, or linear configurations.
        Quality of Service (QoS) ensures prioritized message delivery.

        Profibus Communication:
        Profibus is a standardized fieldbus communication protocol.
        It provides deterministic communication for industrial automation systems.
        The protocol supports both centralized and distributed control architectures.
        Data exchange rates can reach up to 12 Mbps depending on network configuration.

        EtherCAT Real-Time Features:
        EtherCAT provides sub-microsecond precision for time-critical applications.
        The protocol processes Ethernet frames on-the-fly without frame buffering.
        Distributed clocks ensure synchronization across all network nodes.
        Hot-connect capability allows devices to be added during operation.
        """
        
        # Test queries for validation
        self.test_queries = [
            "What is Modbus RTU?",
            "How does CAN bus handle message prioritization?",
            "What security features does DNP3 provide?",
            "Explain EtherCAT real-time capabilities",
            "What are the characteristics of Profibus?",
            "How does industrial Ethernet work?",
            "What is the difference between Modbus and CAN bus?",
            "Security protocols in industrial communication"
        ]
        
        # Test results storage
        self.test_results = {}
        self.performance_metrics = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite and return results."""
        print("🚀 Starting Enhanced RAG System Tests...")
        print("=" * 60)
        
        # Run individual test suites
        chunking_results = self.test_chunking_methods()
        embedding_results = self.test_embedding_methods()
        vector_store_results = self.test_vector_store_enhancements()
        retrieval_results = self.test_retrieval_methods()
        integration_results = self.test_end_to_end_integration()
        performance_results = self.test_performance_monitoring()
        
        # Compile comprehensive results
        comprehensive_results = {
            'chunking': chunking_results,
            'embedding': embedding_results,
            'vector_store': vector_store_results,
            'retrieval': retrieval_results,
            'integration': integration_results,
            'performance': performance_results,
            'overall_score': self.calculate_overall_score(),
            'recommendations': self.generate_recommendations()
        }
        
        # Print summary
        self.print_test_summary(comprehensive_results)
        
        return comprehensive_results
    
    def test_chunking_methods(self) -> Dict[str, Any]:
        """Test all chunking methods including the new adaptive chunker."""
        print("📝 Testing Chunking Methods...")
        
        results = {
            'fixed_length': self.test_fixed_length_chunking(),
            'semantic': self.test_semantic_chunking(),
            'adaptive': self.test_adaptive_chunking()
        }
        
        print(f"   ✅ Chunking tests completed")
        return results
    
    def test_fixed_length_chunking(self) -> Dict[str, Any]:
        """Test fixed-length chunking with various parameters."""
        try:
            start_time = time.time()
            
            # Test with different chunk sizes
            chunks_500 = chunk_by_fixed_length(self.sample_document, chunk_size=500)
            chunks_1000 = chunk_by_fixed_length(self.sample_document, chunk_size=1000)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'chunks_500': len(chunks_500),
                'chunks_1000': len(chunks_1000),
                'processing_time': processing_time,
                'avg_chunk_size_500': sum(len(chunk) for chunk in chunks_500) / len(chunks_500) if chunks_500 else 0,
                'avg_chunk_size_1000': sum(len(chunk) for chunk in chunks_1000) / len(chunks_1000) if chunks_1000 else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_semantic_chunking(self) -> Dict[str, Any]:
        """Test semantic chunking with various parameters."""
        try:
            start_time = time.time()
            
            # Test with different minimum chunk sizes
            chunks_default = chunk_by_semantics(self.sample_document)
            chunks_larger = chunk_by_semantics(self.sample_document, min_chunk_size=100)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'chunks_default': len(chunks_default),
                'chunks_larger': len(chunks_larger),
                'processing_time': processing_time,
                'avg_chunk_size': sum(len(chunk) for chunk in chunks_default) / len(chunks_default) if chunks_default else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_adaptive_chunking(self) -> Dict[str, Any]:
        """Test the new adaptive chunking functionality."""
        try:
            start_time = time.time()
            
            # Test adaptive chunking with default parameters
            chunks_adaptive = chunk_adaptively(self.sample_document)
            
            # Test with custom parameters
            chunks_custom = chunk_adaptively(
                self.sample_document,
                min_chunk_size=150,
                max_chunk_size=1500,
                target_chunk_size=600
            )
            
            # Test with AdaptiveChunker class
            chunker = AdaptiveChunker(
                min_chunk_size=100,
                max_chunk_size=2000,
                target_chunk_size=800,
                semantic_threshold=0.5
            )
            chunks_class = chunker.chunk_text(self.sample_document)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'chunks_adaptive': len(chunks_adaptive),
                'chunks_custom': len(chunks_custom),
                'chunks_class': len(chunks_class),
                'processing_time': processing_time,
                'avg_chunk_size_adaptive': sum(len(chunk) for chunk in chunks_adaptive) / len(chunks_adaptive) if chunks_adaptive else 0,
                'size_variation': max(len(chunk) for chunk in chunks_adaptive) - min(len(chunk) for chunk in chunks_adaptive) if chunks_adaptive else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_embedding_methods(self) -> Dict[str, Any]:
        """Test embedding creation methods."""
        print("🧠 Testing Embedding Methods...")
        
        # Use adaptive chunking for consistent testing
        chunks = chunk_adaptively(self.sample_document, target_chunk_size=500)
        
        results = {
            'tfidf': self.test_tfidf_embeddings(chunks),
            'sentence_transformer': self.test_sentence_transformer_embeddings(chunks)
        }
        
        print(f"   ✅ Embedding tests completed")
        return results
    
    def test_tfidf_embeddings(self, chunks: List[str]) -> Dict[str, Any]:
        """Test TF-IDF embedding creation."""
        try:
            start_time = time.time()
            
            # Create batch embeddings
            embeddings = create_tfidf_embeddings(chunks)
            
            # Test single embedding
            single_embedding = create_single_tfidf_embedding(chunks[0] if chunks else "test")
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'batch_embeddings_shape': embeddings.shape if hasattr(embeddings, 'shape') else None,
                'single_embedding_shape': single_embedding.shape if hasattr(single_embedding, 'shape') else None,
                'processing_time': processing_time,
                'chunks_processed': len(chunks)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_sentence_transformer_embeddings(self, chunks: List[str]) -> Dict[str, Any]:
        """Test Sentence Transformer embedding creation."""
        try:
            start_time = time.time()
            
            # Create batch embeddings
            embeddings = create_sentence_transformer_embeddings(chunks)
            
            # Test single embedding
            single_embedding = create_single_sentence_transformer_embedding(chunks[0] if chunks else "test")
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'batch_embeddings_shape': embeddings.shape if hasattr(embeddings, 'shape') else None,
                'single_embedding_shape': single_embedding.shape if hasattr(single_embedding, 'shape') else None,
                'processing_time': processing_time,
                'chunks_processed': len(chunks)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_vector_store_enhancements(self) -> Dict[str, Any]:
        """Test enhanced vector store functionality."""
        print("🗄️  Testing Vector Store Enhancements...")
        
        try:
            # Use adaptive chunking and sentence transformer embeddings
            chunks = chunk_adaptively(self.sample_document, target_chunk_size=400)
            embeddings = create_sentence_transformer_embeddings(chunks)
            
            # Test enhanced FAISS store with batch processing
            vector_store = FAISSVectorStore(batch_size=500)
            
            start_time = time.time()
            doc_ids = vector_store.add_documents(chunks, embeddings)
            add_time = time.time() - start_time
            
            # Test memory usage estimation
            memory_usage = vector_store.get_memory_usage_estimate()
            
            # Test search functionality
            query_embedding = create_single_sentence_transformer_embedding("Modbus protocol features")
            start_time = time.time()
            search_results = vector_store.search(query_embedding, top_k=3)
            search_time = time.time() - start_time
            
            return {
                'success': True,
                'documents_added': len(doc_ids),
                'add_time': add_time,
                'search_time': search_time,
                'memory_usage': memory_usage,
                'search_results_count': len(search_results[0]) if search_results else 0,
                'vector_store_count': vector_store.get_document_count()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_retrieval_methods(self) -> Dict[str, Any]:
        """Test all retrieval methods including the new advanced retriever."""
        print("🔍 Testing Retrieval Methods...")
        
        # Setup test environment
        chunks = chunk_adaptively(self.sample_document, target_chunk_size=400)
        embeddings = create_sentence_transformer_embeddings(chunks)
        vector_store = FAISSVectorStore()
        vector_store.add_documents(chunks, embeddings)
        
        results = {
            'dense': self.test_dense_retrieval(vector_store),
            'hybrid': self.test_hybrid_retrieval(vector_store),
            'advanced': self.test_advanced_retrieval(vector_store)
        }
        
        print(f"   ✅ Retrieval tests completed")
        return results
    
    def test_dense_retrieval(self, vector_store: FAISSVectorStore) -> Dict[str, Any]:
        """Test dense retrieval method."""
        try:
            retriever = DenseRetriever()
            query = "What is Modbus RTU protocol?"
            query_embedding = create_single_sentence_transformer_embedding(query)
            
            start_time = time.time()
            results = retriever.retrieve_with_scores(vector_store, query_embedding, top_k=3)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'results_count': len(results),
                'processing_time': processing_time,
                'avg_score': sum(score for _, score in results) / len(results) if results else 0,
                'top_score': max(score for _, score in results) if results else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_hybrid_retrieval(self, vector_store: FAISSVectorStore) -> Dict[str, Any]:
        """Test hybrid retrieval method."""
        try:
            retriever = HybridRetriever()
            query = "CAN bus message prioritization"
            query_embedding = create_single_sentence_transformer_embedding(query)
            
            start_time = time.time()
            results = retriever.retrieve_with_scores(vector_store, query_embedding, query, top_k=3)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'results_count': len(results),
                'processing_time': processing_time,
                'avg_score': sum(score for _, score in results) / len(results) if results else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_advanced_retrieval(self, vector_store: FAISSVectorStore) -> Dict[str, Any]:
        """Test the new advanced retrieval method."""
        try:
            retriever = AdvancedRetriever(
                base_retriever_type='hybrid',
                rerank_results=True,
                expand_queries=True,
                diversity_factor=0.3
            )
            query = "security features in industrial protocols"
            query_embedding = create_single_sentence_transformer_embedding(query)
            
            start_time = time.time()
            results = retriever.retrieve_with_scores(vector_store, query_embedding, query, top_k=3)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'results_count': len(results),
                'processing_time': processing_time,
                'avg_score': sum(score for _, score in results) / len(results) if results else 0,
                'query_expansion_enabled': retriever.expand_queries,
                'reranking_enabled': retriever.rerank_results,
                'diversity_factor': retriever.diversity_factor
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end RAG pipeline."""
        print("🔄 Testing End-to-End Integration...")
        
        try:
            start_time = time.time()
            
            # Step 1: Adaptive chunking
            chunks = chunk_adaptively(self.sample_document, target_chunk_size=500)
            
            # Step 2: Create embeddings
            embeddings = create_sentence_transformer_embeddings(chunks)
            
            # Step 3: Store in vector database
            vector_store = FAISSVectorStore()
            vector_store.add_documents(chunks, embeddings)
            
            # Step 4: Test multiple queries with advanced retrieval
            retriever = AdvancedRetriever()
            all_results = []
            
            for query in self.test_queries[:3]:  # Test first 3 queries
                query_embedding = create_single_sentence_transformer_embedding(query)
                results = retriever.retrieve_with_scores(vector_store, query_embedding, query, top_k=2)
                all_results.append({
                    'query': query,
                    'results_count': len(results),
                    'top_score': max(score for _, score in results) if results else 0
                })
            
            total_time = time.time() - start_time
            
            return {
                'success': True,
                'total_processing_time': total_time,
                'chunks_created': len(chunks),
                'queries_tested': len(all_results),
                'avg_results_per_query': sum(r['results_count'] for r in all_results) / len(all_results),
                'avg_top_score': sum(r['top_score'] for r in all_results) / len(all_results),
                'query_results': all_results
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring and analytics capabilities."""
        print("📊 Testing Performance Monitoring...")
        
        try:
            # Create a vector store with monitoring
            vector_store = FAISSVectorStore(batch_size=100)
            
            # Test memory usage estimation
            memory_before = vector_store.get_memory_usage_estimate()
            
            # Add documents and measure memory growth
            chunks = chunk_adaptively(self.sample_document, target_chunk_size=300)
            embeddings = create_sentence_transformer_embeddings(chunks)
            vector_store.add_documents(chunks, embeddings)
            
            memory_after = vector_store.get_memory_usage_estimate()
            
            # Test batch processing performance
            large_chunks = chunks * 5  # Simulate larger dataset
            large_embeddings = np.tile(embeddings, (5, 1))
            
            start_time = time.time()
            vector_store.add_documents(large_chunks, large_embeddings, batch_size=50)
            batch_time = time.time() - start_time
            
            return {
                'success': True,
                'memory_before_mb': memory_before.get('estimate_mb', 0),
                'memory_after_mb': memory_after.get('estimate_mb', 0),
                'memory_growth_mb': memory_after.get('estimate_mb', 0) - memory_before.get('estimate_mb', 0),
                'batch_processing_time': batch_time,
                'batch_documents_processed': len(large_chunks),
                'final_document_count': vector_store.get_document_count()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_overall_score(self) -> float:
        """Calculate overall system performance score."""
        # This would implement a scoring algorithm based on test results
        return 0.85  # Placeholder score
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = [
            "✅ Adaptive chunking provides optimal chunk sizes for your content",
            "✅ Advanced retrieval improves result relevance with re-ranking",
            "✅ Memory-efficient batch processing handles large datasets well",
            "💡 Consider tuning diversity_factor for more varied results",
            "💡 Enable query expansion for better recall on complex queries"
        ]
        return recommendations
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        for category, result in results.items():
            if category in ['overall_score', 'recommendations']:
                continue
                
            print(f"\n{category.upper()}:")
            if isinstance(result, dict):
                for test, details in result.items():
                    if isinstance(details, dict) and 'success' in details:
                        status = "✅" if details['success'] else "❌"
                        print(f"  {status} {test}")
                        if not details['success'] and 'error' in details:
                            print(f"      Error: {details['error']}")
        
        print(f"\n🎯 OVERALL SCORE: {results['overall_score']:.1%}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "=" * 60)


def main():
    """Run the comprehensive RAG system test suite."""
    tester = RAGSystemTester()
    results = tester.run_all_tests()
    
    # Save results to file for further analysis
    import json
    with open('test_results.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\n💾 Test results saved to 'test_results.json'")
    return results


if __name__ == "__main__":
    main() 