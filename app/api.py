"""RAG System Web API - Clean and Simple Interface

This API wraps the existing RAG components to provide a web interface
for document upload, querying, and system configuration.
"""

import os  # Operating system interface for file operations
import logging  # Logging system for debugging and monitoring
import re  # Regular expressions for text pattern matching
from typing import Dict, List, Any, Optional, Tuple  # Type hints for better code clarity
from flask import Flask, request, jsonify, render_template, send_from_directory  # Web framework components
from flask_cors import CORS  # Cross-Origin Resource Sharing for frontend access
import json  # JSON handling for API responses
import traceback  # Error traceback for debugging
import tempfile  # Temporary file handling for uploads
from werkzeug.utils import secure_filename  # Secure filename handling
import numpy as np  # For handling numpy arrays in embedding creation
from datetime import datetime  # For timestamp in health check
import sys  # For system information in health check

# File processing imports
try:
    import PyPDF2  # PDF text extraction
    PDF_AVAILABLE = True  # PDF processing available
except ImportError:
    PDF_AVAILABLE = False  # PDF processing not available

try:
    import pdfplumber  # Advanced PDF parsing
    PDFPLUMBER_AVAILABLE = True  # Advanced PDF processing available
except ImportError:
    PDFPLUMBER_AVAILABLE = False  # Advanced PDF processing not available

# Removed docx import - not needed per user request

# Import our existing RAG components - no changes needed!
from app.chunkers.semantic_chunker import chunk_by_semantics  # Semantic text chunking
from app.chunkers.fixed_length_chunker import chunk_by_fixed_length  # Fixed-length text chunking
from app.chunkers.adaptive_chunker import chunk_adaptively  # NEW: Adaptive chunking
from app.embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding  # TF-IDF embeddings
from app.embedders.sentence_transformer_embedder import create_sentence_transformer_embeddings, create_single_sentence_transformer_embedding  # Neural embeddings
from app.vector_db.faiss_store import FAISSVectorStore  # Vector database for storing embeddings
from app.retriever.dense_retriever import DenseRetriever  # Dense vector retrieval
from app.retriever.hybrid_retriever import HybridRetriever  # Hybrid dense+sparse retrieval
from app.retriever.advanced_retriever import AdvancedRetriever  # NEW: Advanced retrieval with re-ranking
# Ollama support removed - no external dependencies required
from app.augmented_generation.huggingface_generator import create_huggingface_generator  # HuggingFace LLM generator

# Configure logging for monitoring API behavior
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger(__name__)  # Create logger instance for this module

# Initialize Flask application
app = Flask(__name__, 
            static_folder='../web_ui/assets',  # Serve static files from web_ui/assets folder
            template_folder='../web_ui/pages')  # HTML templates from web_ui/pages folder
CORS(app)  # Enable CORS for all routes - allows frontend to access API

# Configure file upload settings
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use system temp directory

# Error handler for file size limit exceeded
@app.errorhandler(413)
def too_large(e):
    """Handle file size too large errors."""
    return jsonify({'error': 'File size exceeds the 50MB limit. Please choose a smaller file.'}), 413

# Global system state - stores current configuration and data
class RAGSystemState:
    """Manages the current state of the RAG system including configuration and data."""
    
    def __init__(self):
        """Initialize RAG system with default configuration."""
        # Enhanced configuration with more options
        self.config = {
            'chunking_method': 'adaptive',  # Use adaptive chunking for better performance
            'embedding_method': 'sentence_transformer',  # Use neural embeddings for better semantic understanding
            'retrieval_method': 'advanced',  # Use advanced retrieval with re-ranking
            'generation_method': 'huggingface',  # Enable AI generation for synthesized answers
            'generation_model': 'google/flan-t5-large',  # Use FLAN-T5-large for better technical synthesis
            # NEW: Advanced configuration options
            'chunk_size': 800,  # Target chunk size for adaptive chunking
            'min_chunk_size': 100,  # Minimum chunk size
            'max_chunk_size': 2000,  # Maximum chunk size
            'retrieval_top_k': 5,  # Number of chunks to retrieve
            'enable_query_expansion': True,  # Enable query expansion in advanced retrieval
            'enable_reranking': True,  # Enable result re-ranking
            'diversity_factor': 0.3,  # Diversity factor for result filtering
            'batch_size': 1000,  # Batch size for processing large datasets
            'enable_analytics': True,  # Enable analytics tracking
            'cache_embeddings': True  # Enable embedding caching
        }
        
        # Log the configuration being loaded for debugging
        logger.info(f"RAG system initialized with enhanced configuration: {self.config}")  # Debug log for config
        
        # System data storage
        self.documents = {}  # Store processed documents by ID
        self.vector_store = FAISSVectorStore(batch_size=self.config['batch_size'])  # Initialize vector database with batch processing
        self.chunk_embeddings = []  # Store all chunk embeddings
        self.all_chunks = []  # Store all text chunks
        self.next_doc_id = 1  # Counter for document IDs
        
        # Cached models to avoid reloading
        self.cached_generator = None  # Cache LLM generator to avoid reloading
        self.cached_generator_type = None  # Track which generator is cached
        
        # NEW: Analytics and monitoring
        self.analytics = {
            'total_documents_processed': 0,  # Total documents ingested
            'total_queries_processed': 0,  # Total queries handled
            'total_chunks_created': 0,  # Total chunks generated
            'average_response_time': 0.0,  # Average query response time
            'last_activity_timestamp': None,  # Last system activity
            'error_count': 0,  # Number of errors encountered
            'retrieval_stats': {  # Retrieval method usage statistics
                'dense': 0,
                'hybrid': 0,
                'advanced': 0
            },
            'chunking_stats': {  # Chunking method usage statistics
                'fixed_length': 0,
                'semantic': 0,
                'adaptive': 0
            },
            'embedding_stats': {  # Embedding method usage statistics
                'tfidf': 0,
                'sentence_transformer': 0
            }
        }
        
        # NEW: Performance monitoring
        self.performance_metrics = {
            'memory_usage': {},  # Memory usage tracking
            'processing_times': [],  # Processing time history
            'cache_hit_rate': 0.0,  # Embedding cache hit rate
            'system_health_score': 1.0  # Overall system health (0-1)
        }

# Initialize global system state
rag_state = RAGSystemState()

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')  # Render main HTML page

@app.route('/settings')
def settings():
    """Serve the settings configuration page."""
    return render_template('settings.html')  # Render settings HTML page

@app.route('/api/health')
def health_check():
    """Check system health and return comprehensive status information."""
    try:
        # Import diagnostic functions
        from app.embedders.sentence_transformer_embedder import get_model_info, reset_model  # Model diagnostics
        from app.embedders.tfidf_embedder import get_vectorizer_info  # Vectorizer diagnostics
        
        # Check if system components are working
        health_status = {
            'status': 'healthy',  # Overall system status - will be updated if issues found
            'timestamp': str(datetime.now()),  # Current timestamp for diagnostics
            'documents_count': len(rag_state.documents),  # Number of processed documents
            'chunks_count': len(rag_state.all_chunks),  # Total chunks in system
            'config': rag_state.config.copy(),  # Current configuration
            'vector_store_ready': rag_state.vector_store is not None,  # Vector DB status
            'diagnostics': {}  # Detailed diagnostic information
        }
        
        # Add component diagnostics
        try:
            # Get embedding model information
            sentence_model_info = get_model_info()  # Get SentenceTransformer status
            health_status['diagnostics']['sentence_transformer'] = sentence_model_info  # Add to diagnostics
            
            # Get TF-IDF vectorizer information
            tfidf_info = get_vectorizer_info()  # Get TF-IDF vectorizer status
            health_status['diagnostics']['tfidf'] = tfidf_info  # Add to diagnostics
            
            # Test embedding creation if possible
            if rag_state.config['embedding_method'] == 'sentence_transformer':
                try:
                    # Try to create a test embedding
                    test_embedding = create_single_sentence_transformer_embedding("Health check test")  # Test embedding
                    health_status['diagnostics']['embedding_test'] = {
                        'status': 'success',  # Test passed
                        'embedding_shape': test_embedding.shape,  # Shape of test embedding
                        'method': 'sentence_transformer'  # Method used
                    }
                except Exception as test_error:
                    # Embedding test failed
                    health_status['status'] = 'degraded'  # Mark as degraded
                    health_status['diagnostics']['embedding_test'] = {
                        'status': 'failed',  # Test failed
                        'error': str(test_error),  # Error details
                        'method': 'sentence_transformer'  # Method that failed
                    }
            else:
                try:
                    # Try TF-IDF test embedding
                    test_embedding = create_single_tfidf_embedding("Health check test")  # Test TF-IDF embedding
                    health_status['diagnostics']['embedding_test'] = {
                        'status': 'success',  # Test passed
                        'embedding_shape': test_embedding.shape,  # Shape of test embedding
                        'method': 'tfidf'  # Method used
                    }
                except Exception as test_error:
                    # TF-IDF test failed
                    health_status['status'] = 'degraded'  # Mark as degraded
                    health_status['diagnostics']['embedding_test'] = {
                        'status': 'failed',  # Test failed
                        'error': str(test_error),  # Error details
                        'method': 'tfidf'  # Method that failed
                    }
            
        except Exception as diag_error:
            # Diagnostics collection failed
            logger.warning(f"Could not collect full diagnostics: {str(diag_error)}")  # Log diagnostic error
            health_status['diagnostics']['collection_error'] = str(diag_error)  # Add error to diagnostics
        
        # Check vector store health
        if rag_state.vector_store is not None:
            try:
                # Get vector store information
                doc_count = rag_state.vector_store.get_document_count()  # Get document count from vector store
                embedding_dim = rag_state.vector_store.get_embedding_dimension()  # Get embedding dimension
                
                health_status['diagnostics']['vector_store'] = {
                    'status': 'healthy',  # Vector store status
                    'document_count': doc_count,  # Documents in vector store
                    'embedding_dimension': embedding_dim,  # Embedding dimension
                    'store_type': 'FAISS'  # Vector store type
                }
            except Exception as vs_error:
                # Vector store check failed
                health_status['status'] = 'degraded'  # Mark as degraded
                health_status['diagnostics']['vector_store'] = {
                    'status': 'error',  # Vector store error
                    'error': str(vs_error)  # Error details
                }
        else:
            # No vector store initialized
            health_status['diagnostics']['vector_store'] = {
                'status': 'not_initialized',  # Vector store not ready
                'message': 'No documents uploaded yet'  # Explanation
            }
        
        # Add system information
        health_status['diagnostics']['system'] = {
            'python_version': sys.version,  # Python version
            'platform': sys.platform,  # Operating system platform
            'available_memory': 'unknown'  # Memory info (would need psutil)
        }
        
        return jsonify(health_status)  # Return comprehensive health status
        
    except Exception as e:
        # Handle any errors in health check
        logger.error(f"Health check failed: {str(e)}")  # Log the error
        logger.error(traceback.format_exc())  # Log full traceback
        return jsonify({
            'status': 'unhealthy',  # Mark system as unhealthy
            'error': str(e),  # Include error message
            'error_type': type(e).__name__,  # Include error type
            'timestamp': str(datetime.now())  # Current timestamp
        }), 500  # Return HTTP 500 status

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current system configuration."""
    return jsonify(rag_state.config)  # Return current config as JSON

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration with user preferences."""
    try:
        # Get new configuration from request
        new_config = request.get_json()  # Parse JSON from request body
        
        # Validate configuration values
        valid_chunking = ['semantic', 'fixed_length']  # Allowed chunking methods
        valid_embedding = ['tfidf', 'sentence_transformer']  # Allowed embedding methods
        valid_retrieval = ['dense', 'hybrid']  # Allowed retrieval methods
        valid_generation = ['huggingface', 'none']  # Allowed generation methods
        
        # Check if all provided values are valid
        if new_config.get('chunking_method') and new_config.get('chunking_method') not in valid_chunking:
            return jsonify({'error': 'Invalid chunking method'}), 400  # Return error for invalid chunking
            
        if new_config.get('embedding_method') and new_config.get('embedding_method') not in valid_embedding:
            return jsonify({'error': 'Invalid embedding method'}), 400  # Return error for invalid embedding
            
        if new_config.get('retrieval_method') and new_config.get('retrieval_method') not in valid_retrieval:
            return jsonify({'error': 'Invalid retrieval method'}), 400  # Return error for invalid retrieval
            
        if new_config.get('generation_method') and new_config.get('generation_method') not in valid_generation:
            return jsonify({'error': 'Invalid generation method'}), 400  # Return error for invalid generation
        
        # Update configuration with valid values
        rag_state.config.update(new_config)  # Merge new config with existing
        
        # Clear cached generator if generation method changed
        if 'generation_method' in new_config or 'generation_model' in new_config:
            rag_state.cached_generator = None  # Force reload of generator
            rag_state.cached_generator_type = None  # Clear generator type cache
        
        logger.info(f"Configuration updated: {rag_state.config}")  # Log configuration change
        
        return jsonify({
            'message': 'Configuration updated successfully',  # Success message
            'config': rag_state.config  # Return updated configuration
        })
        
    except Exception as e:
        # Handle configuration update errors
        logger.error(f"Config update failed: {str(e)}")  # Log the error
        return jsonify({'error': str(e)}), 500  # Return error response

def create_chunks_with_method(text: str, method: str) -> List[str]:
    """Create chunks using the specified chunking method with enhanced options.
    
    Args:
        text: Text to chunk
        method: Chunking method ('fixed_length', 'semantic', 'adaptive')
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If method is not supported
    """
    # Update analytics if enabled
    if rag_state.config.get('enable_analytics', True):
        rag_state.analytics['chunking_stats'][method] = rag_state.analytics['chunking_stats'].get(method, 0) + 1
    
    # Apply chunking method based on configuration
    if method == 'semantic':
        chunks = chunk_by_semantics(text, min_chunk_size=rag_state.config.get('min_chunk_size', 50))
    elif method == 'adaptive':
        # Use adaptive chunking with configuration parameters
        chunks = chunk_adaptively(
            text,
            min_chunk_size=rag_state.config.get('min_chunk_size', 100),
            max_chunk_size=rag_state.config.get('max_chunk_size', 2000),
            target_chunk_size=rag_state.config.get('chunk_size', 800)
        )
    else:  # default to fixed_length
        chunks = chunk_by_fixed_length(text, chunk_size=rag_state.config.get('chunk_size', 500))
    
    # Update chunk statistics
    if rag_state.config.get('enable_analytics', True):
        rag_state.analytics['total_chunks_created'] += len(chunks)
    
    return chunks

def create_embeddings_with_method(chunks: List[str], method: str):
    """Create embeddings using the specified method with fallback handling."""
    # Early validation of input parameters
    if not chunks:
        logger.warning("Empty chunks list provided to create_embeddings_with_method")  # Log empty input
        return np.array([])  # Return empty array for empty chunks
    
    # Log the embedding method being used
    logger.info(f"Creating embeddings using method: {method} for {len(chunks)} chunks")  # Log method and count
    
    try:
        # Choose embedding method based on configuration
        if method == 'tfidf':
            logger.debug("Using TF-IDF embeddings")  # Log TF-IDF method
            return create_tfidf_embeddings(chunks)  # Use TF-IDF for speed
        else:
            logger.debug("Using SentenceTransformer embeddings")  # Log neural method
            return create_sentence_transformer_embeddings(chunks)  # Use neural embeddings for quality
            
    except Exception as e:
        # Log the specific error details
        logger.error(f"Error creating embeddings with {method}: {str(e)}")  # Log embedding error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        
        # If SentenceTransformer fails, fall back to TF-IDF
        if method == 'sentence_transformer':
            logger.warning("SentenceTransformer failed, falling back to TF-IDF embeddings")  # Log fallback
            try:
                # Update configuration to reflect fallback
                rag_state.config['embedding_method'] = 'tfidf'  # Update config to TF-IDF
                return create_tfidf_embeddings(chunks)  # Use TF-IDF as fallback
            except Exception as fallback_error:
                logger.error(f"Fallback to TF-IDF also failed: {str(fallback_error)}")  # Log fallback failure
                raise RuntimeError(f"Both embedding methods failed. Primary: {str(e)}, Fallback: {str(fallback_error)}")  # Raise combined error
        else:
            # TF-IDF failed, no fallback available
            logger.error("TF-IDF embedding creation failed with no fallback available")  # Log TF-IDF failure
            raise RuntimeError(f"TF-IDF embedding creation failed: {str(e)}")  # Re-raise the original error

def create_query_embedding_with_method(query: str, method: str):
    """Create query embedding using the specified method with fallback handling."""
    # Early validation of input parameters
    if not isinstance(query, str):
        logger.error(f"Query must be string, got {type(query)}")  # Log type error
        raise ValueError(f"Query must be string, got {type(query)}")  # Raise type error
    
    if not query or not query.strip():
        logger.warning("Empty query provided to create_query_embedding_with_method")  # Log empty query
        raise ValueError("Query cannot be empty")  # Raise error for empty query
    
    # Clean the query text
    clean_query = query.strip()  # Remove leading/trailing whitespace
    
    # Log the embedding method being used
    logger.info(f"Creating query embedding using method: {method}")  # Log method
    
    try:
        # Choose query embedding method based on configuration
        if method == 'tfidf':
            logger.debug("Using TF-IDF for query embedding")  # Log TF-IDF method
            return create_single_tfidf_embedding(clean_query)  # Use TF-IDF for consistency
        else:
            logger.debug("Using SentenceTransformer for query embedding")  # Log neural method
            return create_single_sentence_transformer_embedding(clean_query)  # Use neural embeddings
            
    except Exception as e:
        # Log the specific error details
        logger.error(f"Error creating query embedding with {method}: {str(e)}")  # Log embedding error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        
        # If SentenceTransformer fails, fall back to TF-IDF
        if method == 'sentence_transformer':
            logger.warning("SentenceTransformer failed for query, falling back to TF-IDF")  # Log fallback
            try:
                # Update configuration to reflect fallback
                rag_state.config['embedding_method'] = 'tfidf'  # Update config to TF-IDF
                return create_single_tfidf_embedding(clean_query)  # Use TF-IDF as fallback
            except Exception as fallback_error:
                logger.error(f"Fallback to TF-IDF also failed for query: {str(fallback_error)}")  # Log fallback failure
                raise RuntimeError(f"Both query embedding methods failed. Primary: {str(e)}, Fallback: {str(fallback_error)}")  # Raise combined error
        else:
            # TF-IDF failed, no fallback available
            logger.error("TF-IDF query embedding creation failed with no fallback available")  # Log TF-IDF failure
            raise RuntimeError(f"TF-IDF query embedding creation failed: {str(e)}")  # Re-raise the original error

def create_retriever_with_method(method: str):
    """Create retriever instance based on configured method.
    
    Args:
        method: Retrieval method ('dense', 'hybrid', 'advanced')
        
    Returns:
        Retriever instance
    """
    # Update analytics if enabled
    if rag_state.config.get('enable_analytics', True):
        rag_state.analytics['retrieval_stats'][method] = rag_state.analytics['retrieval_stats'].get(method, 0) + 1
    
    # Create retriever based on method
    if method == 'hybrid':
        return HybridRetriever()
    elif method == 'advanced':
        # Create advanced retriever with configuration
        return AdvancedRetriever(
            base_retriever_type='hybrid',
            rerank_results=rag_state.config.get('enable_reranking', True),
            expand_queries=rag_state.config.get('enable_query_expansion', True),
            diversity_factor=rag_state.config.get('diversity_factor', 0.3)
        )
    else:  # default to dense
        return DenseRetriever()

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded files based on file type."""
    import time  # Import time for timing logs
    start_time = time.time()  # Start timing
    
    # Get file extension to determine processing method
    file_extension = filename.lower().split('.')[-1]  # Get lowercase file extension
    
    logger.info(f"Starting text extraction from {filename} ({file_extension})")  # Log extraction start
    
    try:
        if file_extension == 'txt':
            # Process plain text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()  # Return file contents directly
                logger.info(f"Text file extracted in {time.time() - start_time:.2f} seconds")  # Log timing
                return text
                
        elif file_extension == 'pdf':
            # Process PDF files - use PyPDF2 first for speed
            if PDF_AVAILABLE:
                # Use PyPDF2 (faster for most PDFs)
                logger.info("Using PyPDF2 for fast PDF extraction")  # Log method
                text = ""  # Initialize text accumulator
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)  # Create PDF reader
                    for page_num, page in enumerate(pdf_reader.pages):  # Iterate through pages
                        try:
                            page_text = page.extract_text()  # Extract text from page
                            if page_text:
                                text += page_text + "\n"  # Add page text with newline
                        except Exception as page_error:
                            logger.warning(f"Failed to extract page {page_num}: {str(page_error)}")  # Log page errors
                            continue  # Skip problematic pages
                    
                extraction_time = time.time() - start_time  # Calculate extraction time
                logger.info(f"PyPDF2 extraction completed in {extraction_time:.2f} seconds, {len(text)} characters")  # Log results
                return text  # Return extracted text
                
            elif PDFPLUMBER_AVAILABLE:
                # Fallback to pdfplumber (slower but better quality)
                logger.info("Using pdfplumber as fallback")  # Log fallback
                import pdfplumber  # Import pdfplumber library
                text = ""  # Initialize text accumulator
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:  # Iterate through PDF pages
                        page_text = page.extract_text()  # Extract text from page
                        if page_text:
                            text += page_text + "\n"  # Add page text with newline
                extraction_time = time.time() - start_time  # Calculate extraction time
                logger.info(f"pdfplumber extraction completed in {extraction_time:.2f} seconds")  # Log results
                return text  # Return extracted text
            else:
                raise Exception("PDF processing libraries not available. Install PyPDF2 or pdfplumber.")
                
        elif file_extension == 'docx':
            # Process Word documents
            # Removed docx import, so this block will now raise an error
            raise Exception("Word document processing not available. Install python-docx.")
        else:
            raise Exception(f"Unsupported file type: .{file_extension}")
            
    except Exception as e:
        # Handle file processing errors
        extraction_time = time.time() - start_time  # Calculate time even on error
        logger.error(f"Error extracting text from {filename} after {extraction_time:.2f} seconds: {str(e)}")  # Log the error with timing
        raise  # Re-raise the exception for handling upstream

def validate_chunk_relevance(chunks_with_scores: List[Tuple[str, float]], query: str) -> List[Tuple[str, float]]:
    """Validate and filter chunks for relevance and quality.
    
    Args:
        chunks_with_scores: List of (chunk_text, score) tuples
        query: Original user query
        
    Returns:
        List of validated (chunk_text, score) tuples
    """
    if not chunks_with_scores:
        return []  # Return empty list if no chunks
    
    validated_chunks = []  # List for validated chunks
    query_lower = query.lower()  # Lowercase query for matching
    
    # Extract key terms from query
    query_words = set(re.findall(r'\b\w+\b', query_lower))  # Extract words from query
    query_words = {word for word in query_words if len(word) > 2}  # Filter short words
    
    for chunk, score in chunks_with_scores:
        # Skip empty or very short chunks
        if not chunk or len(chunk.strip()) < 50:
            continue  # Skip too short chunks
        
        chunk_lower = chunk.lower()  # Lowercase chunk for matching
        
        # Check for minimum relevance indicators
        relevance_score = 0.0  # Initialize relevance score
        
        # 1. Check for query word matches
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))  # Extract words from chunk
        matching_words = query_words.intersection(chunk_words)  # Find common words
        if query_words:
            word_match_ratio = len(matching_words) / len(query_words)  # Calculate match ratio
            relevance_score += word_match_ratio * 0.4  # Word matching contributes 40%
        
        # 2. Check for exact phrase matches
        if len(query.split()) > 1:  # Multi-word query
            if query_lower in chunk_lower:
                relevance_score += 0.3  # Exact phrase match bonus
        
        # 3. Check for procedural content if query is procedural
        is_procedural_query = any(keyword in query_lower for keyword in [
            'step', 'steps', 'how to', 'install', 'setup', 'procedure', 'instructions'
        ])
        
        if is_procedural_query:
            # Boost score for procedural content
            procedural_indicators = [
                r'\bstep\s*\d+', r'\d+\.\s', r'\binstall', r'\bconnect', 
                r'\bmount', r'\bprocedure', r'\binstructions'
            ]
            
            procedural_matches = 0
            for pattern in procedural_indicators:
                procedural_matches += len(re.findall(pattern, chunk_lower))
            
            if procedural_matches > 0:
                relevance_score += min(0.3, procedural_matches * 0.1)  # Procedural bonus
        
        # 4. Penalize chunks that are just metadata/titles
        metadata_patterns = [
            r'^[A-Z\s\d\-_]+$',  # All caps titles
            r'publication\s+\d+',  # Publication numbers
            r'^\w+\s+\d+[\w\s]*$',  # Simple title patterns
        ]
        
        is_metadata = any(re.match(pattern, chunk.strip()) for pattern in metadata_patterns)
        if is_metadata and len(chunk.strip()) < 200:
            relevance_score *= 0.3  # Heavy penalty for metadata chunks
        
        # 5. Check minimum content quality
        has_meaningful_content = (
            len(chunk.strip()) >= 100 and  # Reasonable length
            '.' in chunk and  # Contains sentences
            any(char.islower() for char in chunk)  # Not all caps
        )
        
        if not has_meaningful_content:
            relevance_score *= 0.5  # Penalty for low-quality content
        
        # Apply minimum relevance threshold
        min_relevance = 0.15  # Minimum relevance threshold
        combined_score = max(score * 0.7 + relevance_score * 0.3, relevance_score)  # Combine scores
        
        if combined_score >= min_relevance:
            validated_chunks.append((chunk, combined_score))  # Add validated chunk
    
    # Sort by combined score
    validated_chunks.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    
    # Log validation results
    logger.info(f"Chunk validation: {len(chunks_with_scores)} → {len(validated_chunks)} chunks passed")
    
    return validated_chunks  # Return validated chunks

def get_generator():
    """Get or create LLM generator based on current configuration."""
    # Check if we need to create/update the generator
    current_method = rag_state.config['generation_method']  # Get current generation method
    current_model = rag_state.config.get('generation_model', 'llama2')  # Get current model
    
    # Return None if generation is disabled
    if current_method == 'none':
        return None  # No generation requested
    
    # Check if we can reuse cached generator
    if (rag_state.cached_generator is not None and 
        rag_state.cached_generator_type == f"{current_method}_{current_model}"):
        return rag_state.cached_generator  # Return cached generator
    
    # Create new generator based on method
    try:
        if current_method == 'huggingface':
            generator = create_huggingface_generator(
                "google/flan-t5-base", use_small_model=True)  # Create HuggingFace generator
            
            # Cache the generator for future use
            rag_state.cached_generator = generator  # Store generator in cache
            rag_state.cached_generator_type = f"{current_method}_{current_model}"  # Store generator type
            
            return generator  # Return new generator
        else:
            # No generation method or unsupported method
            return None  # Return None for no generation
        
    except Exception as e:
        # Handle generator creation errors
        logger.error(f"Failed to create generator: {str(e)}")  # Log the error
        return None  # Return None if generator creation fails

@app.route('/api/documents/ingest', methods=['POST'])
def ingest_document():
    """Process and store a new document in the RAG system."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400  # Return error for non-JSON request
            
        # Get document data from request
        data = request.get_json()  # Parse JSON request body
        
        # Validate JSON data
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400  # Return error for invalid JSON
            
        text = data.get('text', '').strip()  # Extract text content
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400  # Return error for empty text
        
        # Create document ID
        doc_id = f"doc_{rag_state.next_doc_id}"  # Generate unique document ID
        rag_state.next_doc_id += 1  # Increment counter for next document
        
        # Process document using current configuration
        chunks = create_chunks_with_method(text, rag_state.config['chunking_method'])  # Chunk the text
        chunk_embeddings = create_embeddings_with_method(chunks, rag_state.config['embedding_method'])  # Create embeddings
        
        # Store document information
        rag_state.documents[doc_id] = {
            'text': text,  # Original document text
            'chunks': chunks,  # Text chunks
            'chunk_count': len(chunks),  # Number of chunks
            'processed_with': rag_state.config.copy()  # Configuration used for processing
        }
        
        # Add chunks to global storage
        rag_state.all_chunks.extend(chunks)  # Add chunks to global list
        
        # Add embeddings to vector store
        rag_state.vector_store.add_documents(chunks, chunk_embeddings)
        
        # Update analytics if enabled
        if rag_state.config.get('enable_analytics', True):
            rag_state.analytics['total_documents_processed'] += 1
            rag_state.analytics['last_activity_timestamp'] = str(datetime.now())
        
        # Log successful processing
        logger.info(f"Document {doc_id} successfully processed: {len(chunks)} chunks created")
        
        # Return success response with detailed information
        return jsonify({
            'success': True,  # Indicate successful processing
            'message': f'Document processed successfully with {len(chunks)} chunks',  # Success message
            'document_id': doc_id,  # Generated document ID
            'chunks_created': len(chunks),  # Number of chunks created
            'embedding_method': rag_state.config['embedding_method'],  # Embedding method used
            'chunking_method': rag_state.config['chunking_method'],  # Chunking method used
            'total_documents': len(rag_state.documents),  # Total documents in system
            'total_chunks': len(rag_state.all_chunks)  # Total chunks in system
        }), 200  # HTTP 200 OK status
        
    except Exception as e:
        # Handle document ingestion errors
        logger.error(f"Document ingestion failed: {str(e)}")  # Log the error
        logger.error(traceback.format_exc())  # Log full traceback for debugging
        return jsonify({'error': str(e)}), 500  # Return error response

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Process user query and return AI-generated response with sources."""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400  # Return error for non-JSON request
            
        # Get query data from request
        data = request.get_json()  # Parse JSON request body
        
        # Validate JSON data
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400  # Return error for invalid JSON
            
        query = data.get('query', '').strip()  # Extract user query
        top_k = data.get('top_k', 5)  # Number of chunks to retrieve
        use_generation = data.get('use_generation', True)  # Whether to generate AI response
        
        # Validate input
        if not query:
            return jsonify({'error': 'No query provided'}), 400  # Return error for empty query
            
        if not rag_state.all_chunks:
            return jsonify({'error': 'No documents have been ingested yet'}), 400  # Return error if no documents
        
        # Create query embedding
        query_embedding = create_query_embedding_with_method(query, rag_state.config['embedding_method'])  # Embed the query
        
        # Retrieve relevant chunks using configured method
        retriever = create_retriever_with_method(rag_state.config['retrieval_method'])
        
        # Handle different retriever types properly
        chunks_with_scores = []  # Initialize with empty list to prevent None iteration
        
        try:
            if rag_state.config['retrieval_method'] == 'advanced':
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, query, top_k)  # Advanced retriever needs query text
            elif rag_state.config['retrieval_method'] == 'hybrid':
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, query, top_k)  # Hybrid retriever needs query text
            else:
                chunks_with_scores = retriever.retrieve_with_scores(
                    rag_state.vector_store, query_embedding, top_k)  # Dense retriever only needs embedding
            
            # Ensure we have a valid list (not None)
            if chunks_with_scores is None:
                chunks_with_scores = []
                logger.warning("Retriever returned None, using empty list")
                
        except Exception as retrieval_error:
            logger.error(f"Retrieval failed: {str(retrieval_error)}")
            chunks_with_scores = []  # Use empty list if retrieval fails
        
        # Update analytics
        if rag_state.config.get('enable_analytics', True):
            rag_state.analytics['total_queries_processed'] += 1
            rag_state.analytics['last_activity_timestamp'] = str(datetime.now())
        
        # Validate and filter chunks for quality
        validated_chunks = validate_chunk_relevance(chunks_with_scores, query)
        
        # Prepare response data
        response_data = {
            'chunks': [chunk for chunk, _ in validated_chunks],  # List of validated text chunks
            'scores': [float(score) for _, score in validated_chunks],  # List of relevance scores
            'num_results': len(validated_chunks),  # Number of results found
            'retrieval_method': rag_state.config['retrieval_method'],  # Method used for retrieval
            'generation_enabled': use_generation,  # Whether generation was requested
            'query_processed': query,  # Echo back the processed query
            'validation_applied': True,  # Indicate validation was applied
            'original_results_count': len(chunks_with_scores)  # Original number before validation
        }
        
        # Generate AI response if requested and possible
        if use_generation and rag_state.config['generation_method'] != 'none':
            generator = get_generator()  # Get LLM generator
            
            if generator:
                # Extract just the chunk texts for generation (use validated chunks)
                chunk_texts = [chunk for chunk, _ in validated_chunks]  # Get validated chunk texts
                
                # Generate response using LLM
                generation_result = generator.generate_response(query, chunk_texts)  # Generate AI response
                
                if generation_result.get('success'):
                    response_data['generated_response'] = generation_result['response']  # Add generated response
                    response_data['model_used'] = generation_result.get('model_used', 'unknown')  # Add model info
                else:
                    response_data['generation_error'] = generation_result.get('error', 'Unknown error')  # Add error info
            else:
                response_data['generation_error'] = 'No generator available'  # Generator not available
        
        logger.info(f"Query processed: '{query[:50]}...' -> {len(chunks_with_scores)} chunks")  # Log query processing
        
        return jsonify(response_data)  # Return query results
        
    except Exception as e:
        # Handle query processing errors
        logger.error(f"Query processing failed: {str(e)}")  # Log the error
        logger.error(traceback.format_exc())  # Log full traceback for debugging
        return jsonify({'error': str(e)}), 500  # Return error response

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Upload and process document files (.txt, .pdf, .docx)."""
    file_obj = None
    temp_file_path = None
    
    try:
        # Debug logging
        logger.info(f"Upload request received - Content-Type: {request.content_type}")
        
        # Check content type more permissively
        content_type = request.content_type or ''
        if 'multipart' not in content_type.lower():
            return jsonify({'error': 'Request must be multipart/form-data'}), 400
        
        # Get uploaded file simply
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file_obj = request.files['file']
        
        if not file_obj or not file_obj.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content into memory first to check size
        file_obj.seek(0)  # Ensure we're at the beginning
        file_content = file_obj.read()  # Read entire file into memory
        file_size = len(file_content)  # Get actual file size
        
        logger.info(f"File received: {file_obj.filename}, size: {file_size} bytes")
        
        # Check if file is empty
        if file_size == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        # Validate file extension
        allowed_extensions = ['txt', 'pdf', 'docx']  # Allowed file types
        file_extension = file_obj.filename.lower().split('.')[-1]  # Get file extension
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type. Please upload {", ".join(allowed_extensions)} files.'}), 400
        
        # Create secure filename
        filename = secure_filename(file_obj.filename)  # Secure the filename
        
        # Save file content to temporary file
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file_path = temp_file.name
            temp_file.write(file_content)  # Write content to temporary file
            temp_file.close()  # Close the file handle
            logger.info(f"File saved to: {temp_file_path} ({file_size} bytes)")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Could not save uploaded file: {str(e)}'}), 500
        
        try:
            # Extract text from uploaded file
            text = extract_text_from_file(temp_file_path, filename)  # Extract text based on file type
            
            # Validate extracted text
            if not text or not text.strip():
                return jsonify({'error': 'No text could be extracted from the file'}), 400
            
            # Create document ID
            doc_id = f"doc_{rag_state.next_doc_id}"  # Generate unique document ID
            rag_state.next_doc_id += 1  # Increment counter for next document
            
            # Process document using current configuration
            chunks = create_chunks_with_method(text, rag_state.config['chunking_method'])  # Chunk the text
            chunk_embeddings = create_embeddings_with_method(chunks, rag_state.config['embedding_method'])  # Create embeddings
            
            # Store document information
            rag_state.documents[doc_id] = {
                'text': text,  # Extracted document text
                'chunks': chunks,  # Text chunks
                'chunk_count': len(chunks),  # Number of chunks
                'filename': filename,  # Original filename
                'file_type': file_extension,  # File extension
                'processed_with': rag_state.config.copy()  # Configuration used for processing
            }
            
            # Add chunks to global storage
            rag_state.all_chunks.extend(chunks)  # Add chunks to global list
            
            # Add embeddings to vector store
            rag_state.vector_store.add_documents(chunks, chunk_embeddings)  # Store in vector database
            
            logger.info(f"File {filename} processed as {doc_id}: {len(chunks)} chunks")  # Log successful processing
            
            return jsonify({
                'message': 'File processed successfully',  # Success message
                'document_id': doc_id,  # Return document ID
                'filename': filename,  # Original filename
                'chunks_created': len(chunks),  # Number of chunks created
                'file_type': file_extension,  # File type processed
                'config_used': rag_state.config.copy()  # Configuration used
            })
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)  # Delete temporary file
                    logger.info(f"Cleaned up temp file: {temp_file_path}")
                except OSError as e:
                    logger.warning(f"Could not clean up temp file {temp_file_path}: {str(e)}")  # Log cleanup errors
        
    except Exception as e:
        # Handle file upload errors
        logger.error(f"File upload failed: {str(e)}")  # Log the error
        logger.error(traceback.format_exc())  # Log full traceback for debugging
        
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)  # Delete temporary file
                logger.info(f"Cleaned up temp file after error: {temp_file_path}")
            except OSError:
                pass  # Ignore cleanup errors in exception handler
        
        # Return more specific error message
        if "ClientDisconnected" in str(e):
            return jsonify({'error': 'File upload was interrupted. Please try again with a smaller file.'}), 400
        elif "Bad Request" in str(e):
            return jsonify({'error': 'Invalid file upload request. Please ensure you selected a valid file.'}), 400
        else:
            return jsonify({'error': f'File processing failed: {str(e)}'}), 500  # Return error response

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Get list of all processed documents."""
    try:
        # Prepare document list with metadata
        document_list = []  # List to store document information
        
        for doc_id, doc_info in rag_state.documents.items():
            document_summary = {
                'id': doc_id,  # Document identifier
                'chunk_count': doc_info['chunk_count'],  # Number of chunks created
                'text_preview': doc_info['text'][:200] + "..." if len(doc_info['text']) > 200 else doc_info['text'],  # Text preview
                'processed_with': doc_info['processed_with'],  # Processing configuration used
                'size_bytes': len(doc_info['text'].encode('utf-8'))  # Document size in bytes
            }
            document_list.append(document_summary)  # Add to document list
        
        # Prepare response with system statistics
        response_data = {
            'documents': document_list,  # List of processed documents
            'total_documents': len(rag_state.documents),  # Total number of documents
            'total_chunks': len(rag_state.all_chunks),  # Total number of chunks
            'vector_store_count': rag_state.vector_store.get_document_count(),  # Vector store document count
            'current_config': rag_state.config.copy()  # Current system configuration
        }
        
        return jsonify(response_data), 200  # Return successful response
        
    except Exception as e:
        # Handle errors in document listing
        logger.error(f"Error listing documents: {str(e)}")  # Log error details
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500  # Return error response


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive system analytics and performance metrics."""
    try:
        # Update real-time metrics
        memory_usage = rag_state.vector_store.get_memory_usage_estimate()  # Get current memory usage
        
        # Calculate performance metrics
        health_score = 1.0  # Start with perfect health
        
        # Adjust health based on error rate
        total_operations = (rag_state.analytics['total_documents_processed'] + 
                          rag_state.analytics['total_queries_processed'])
        if total_operations > 0:
            error_rate = rag_state.analytics['error_count'] / total_operations
            health_score -= min(error_rate * 2, 0.5)  # Reduce health by up to 50% based on errors
        
        # Update health score
        rag_state.performance_metrics['system_health_score'] = max(0.0, health_score)
        rag_state.performance_metrics['memory_usage'] = memory_usage
        
        # Prepare comprehensive analytics response
        analytics_data = {
            'system_overview': {
                'total_documents_processed': rag_state.analytics['total_documents_processed'],
                'total_queries_processed': rag_state.analytics['total_queries_processed'],
                'total_chunks_created': rag_state.analytics['total_chunks_created'],
                'error_count': rag_state.analytics['error_count'],
                'last_activity': rag_state.analytics['last_activity_timestamp'],
                'system_health_score': rag_state.performance_metrics['system_health_score']
            },
            'usage_statistics': {
                'retrieval_methods': rag_state.analytics['retrieval_stats'],
                'chunking_methods': rag_state.analytics['chunking_stats'],
                'embedding_methods': rag_state.analytics['embedding_stats']
            },
            'performance_metrics': {
                'memory_usage': rag_state.performance_metrics['memory_usage'],
                'average_response_time': rag_state.analytics['average_response_time'],
                'cache_hit_rate': rag_state.performance_metrics['cache_hit_rate']
            },
            'configuration': {
                'current_config': rag_state.config.copy(),
                'analytics_enabled': rag_state.config.get('enable_analytics', True),
                'caching_enabled': rag_state.config.get('cache_embeddings', True)
            },
            'recommendations': []  # System optimization recommendations
        }
        
        # Generate performance recommendations
        recommendations = []
        
        # Memory usage recommendations
        if memory_usage.get('estimate_mb', 0) > 1000:  # Over 1GB
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'message': 'High memory usage detected. Consider using batch processing or reducing chunk sizes.',
                'action': 'Increase batch_size in configuration or use smaller chunks'
            })
        
        # Error rate recommendations
        if total_operations > 10 and (rag_state.analytics['error_count'] / total_operations) > 0.1:
            recommendations.append({
                'type': 'reliability',
                'priority': 'high',
                'message': 'High error rate detected. Check system logs for issues.',
                'action': 'Review error logs and consider adjusting configuration'
            })
        
        # Usage pattern recommendations
        retrieval_stats = rag_state.analytics['retrieval_stats']
        if sum(retrieval_stats.values()) > 50:  # Significant usage
            most_used = max(retrieval_stats.keys(), key=lambda k: retrieval_stats[k])
            if most_used == 'dense' and retrieval_stats['dense'] > retrieval_stats['advanced'] * 2:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'message': 'Consider using advanced retrieval for better results.',
                    'action': 'Switch retrieval_method to "advanced" in configuration'
                })
        
        analytics_data['recommendations'] = recommendations
        
        return jsonify(analytics_data), 200  # Return analytics data
        
    except Exception as e:
        # Handle errors in analytics generation
        logger.error(f"Error generating analytics: {str(e)}")  # Log error details
        return jsonify({'error': f'Failed to generate analytics: {str(e)}'}), 500  # Return error response


@app.route('/api/analytics/reset', methods=['POST'])
def reset_analytics():
    """Reset analytics counters (useful for testing or clean start)."""
    try:
        # Reset all analytics counters
        rag_state.analytics = {
            'total_documents_processed': 0,
            'total_queries_processed': 0,
            'total_chunks_created': 0,
            'average_response_time': 0.0,
            'last_activity_timestamp': None,
            'error_count': 0,
            'retrieval_stats': {'dense': 0, 'hybrid': 0, 'advanced': 0},
            'chunking_stats': {'fixed_length': 0, 'semantic': 0, 'adaptive': 0},
            'embedding_stats': {'tfidf': 0, 'sentence_transformer': 0}
        }
        
        # Reset performance metrics
        rag_state.performance_metrics = {
            'memory_usage': {},
            'processing_times': [],
            'cache_hit_rate': 0.0,
            'system_health_score': 1.0
        }
        
        logger.info("Analytics counters reset successfully")
        
        return jsonify({
            'message': 'Analytics counters reset successfully',
            'reset_timestamp': str(datetime.now())
        }), 200
        
    except Exception as e:
        logger.error(f"Error resetting analytics: {str(e)}")
        return jsonify({'error': f'Failed to reset analytics: {str(e)}'}), 500


if __name__ == '__main__':
    """Run the Flask development server."""
    print("🚀 Starting RAG API Server...")  # Startup message
    print("📝 Upload documents and ask questions at: http://localhost:5000")  # Usage instructions
    print("⚙️  Change settings at: http://localhost:5000/settings")  # Settings page info
    
    app.run(debug=True, host='0.0.0.0', port=5000)  # Start development server 