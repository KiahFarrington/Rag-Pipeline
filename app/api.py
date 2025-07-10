"""RAG System Web API - Clean and Simple Interface

This API wraps the existing RAG components to provide a web interface
for document upload, querying, and system configuration.
"""

import os  # Operating system interface for file operations
import logging  # Logging system for debugging and monitoring
from typing import Dict, List, Any, Optional  # Type hints for better code clarity
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

try:
    from docx import Document  # Word document processing
    DOCX_AVAILABLE = True  # Word document processing available
except ImportError:
    DOCX_AVAILABLE = False  # Word document processing not available

# Import our existing RAG components - no changes needed!
from app.chunkers.semantic_chunker import chunk_by_semantics  # Semantic text chunking
from app.chunkers.fixed_length_chunker import chunk_by_fixed_length  # Fixed-length text chunking
from app.embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding  # TF-IDF embeddings
from app.embedders.sentence_transformer_embedder import create_sentence_transformer_embeddings, create_single_sentence_transformer_embedding  # Neural embeddings
from app.vector_db.faiss_store import FAISSVectorStore  # Vector database for storing embeddings
from app.retriever.dense_retriever import DenseRetriever  # Dense vector retrieval
from app.retriever.hybrid_retriever import HybridRetriever  # Hybrid dense+sparse retrieval
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
        # Default configuration - good balance of speed and quality
        self.config = {
            'chunking_method': 'fixed_length',  # Use fixed length chunking for better PDF handling
            'embedding_method': 'tfidf',  # Use TF-IDF for much faster processing
            'retrieval_method': 'dense',  # Default to dense retrieval for simplicity
            'generation_method': 'huggingface',  # Enable AI generation for synthesized answers
            'generation_model': 'google/flan-t5-base'  # Use FLAN-T5 for technical Q&A
        }
        
        # Log the configuration being loaded for debugging
        logger.info(f"RAG system initialized with configuration: {self.config}")  # Debug log for config
        
        # System data storage
        self.documents = {}  # Store processed documents by ID
        self.vector_store = FAISSVectorStore()  # Initialize vector database
        self.chunk_embeddings = []  # Store all chunk embeddings
        self.all_chunks = []  # Store all text chunks
        self.next_doc_id = 1  # Counter for document IDs
        
        # Cached models to avoid reloading
        self.cached_generator = None  # Cache LLM generator to avoid reloading
        self.cached_generator_type = None  # Track which generator is cached

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
    """Create text chunks using the specified method."""
    # Log the chunking method being used for debugging
    logger.info(f"Using chunking method: {method}")  # Debug log to see actual method
    
    # Choose chunking method based on configuration
    if method == 'semantic':
        logger.info("Performing semantic chunking")  # Debug log for semantic
        return chunk_by_semantics(text)  # Use semantic chunking for better context
    else:
        logger.info("Performing fixed-length chunking")  # Debug log for fixed-length
        return chunk_by_fixed_length(text)  # Use fixed-length chunking for speed

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
            if DOCX_AVAILABLE:
                doc = Document(file_path)  # Open Word document
                text = ""  # Initialize text accumulator
                for paragraph in doc.paragraphs:  # Iterate through paragraphs
                    text += paragraph.text + "\n"  # Add paragraph text with newline
                logger.info(f"DOCX file extracted in {time.time() - start_time:.2f} seconds")  # Log timing
                return text  # Return extracted text
            else:
                raise Exception("Word document processing not available. Install python-docx.")
        else:
            raise Exception(f"Unsupported file type: .{file_extension}")
            
    except Exception as e:
        # Handle file processing errors
        extraction_time = time.time() - start_time  # Calculate time even on error
        logger.error(f"Error extracting text from {filename} after {extraction_time:.2f} seconds: {str(e)}")  # Log the error with timing
        raise  # Re-raise the exception for handling upstream

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
        rag_state.vector_store.add_documents(chunks, chunk_embeddings)  # Store in vector database
        
        logger.info(f"Document {doc_id} ingested: {len(chunks)} chunks")  # Log successful ingestion
        
        return jsonify({
            'message': 'Document processed successfully',  # Success message
            'document_id': doc_id,  # Return document ID
            'chunks_created': len(chunks),  # Number of chunks created
            'config_used': rag_state.config.copy()  # Configuration used
        })
        
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
        if rag_state.config['retrieval_method'] == 'hybrid':
            retriever = HybridRetriever()  # Use hybrid retrieval
            chunks_with_scores = retriever.retrieve_with_scores(
                rag_state.vector_store, query_embedding, query, top_k)  # Get chunks with scores
        else:
            retriever = DenseRetriever()  # Use dense retrieval
            chunks_with_scores = retriever.retrieve_with_scores(
                rag_state.vector_store, query_embedding, top_k)  # Get chunks with scores
        
        # Prepare response data
        response_data = {
            'query': query,  # Original user query
            'retrieved_chunks': [],  # List of retrieved chunks
            'config_used': rag_state.config.copy()  # Configuration used
        }
        
        # Format retrieved chunks with scores
        for chunk, score in chunks_with_scores:
            response_data['retrieved_chunks'].append({
                'text': chunk,  # Chunk text content
                'similarity_score': float(score)  # Similarity score
            })
        
        # Generate AI response if requested and possible
        if use_generation and rag_state.config['generation_method'] != 'none':
            generator = get_generator()  # Get LLM generator
            
            if generator:
                # Extract just the chunk texts for generation
                chunk_texts = [chunk for chunk, _ in chunks_with_scores]  # Get chunk texts
                
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
        documents = []  # List to store document information
        
        for doc_id, doc_data in rag_state.documents.items():
            documents.append({
                'id': doc_id,  # Document ID
                'chunk_count': doc_data['chunk_count'],  # Number of chunks
                'text_preview': doc_data['text'][:100] + '...' if len(doc_data['text']) > 100 else doc_data['text'],  # Text preview
                'filename': doc_data.get('filename', 'Text Input'),  # Filename or default
                'file_type': doc_data.get('file_type', 'text'),  # File type or default
                'processed_with': doc_data['processed_with']  # Processing configuration
            })
        
        return jsonify({
            'documents': documents,  # List of documents
            'total_count': len(documents)  # Total document count
        })
        
    except Exception as e:
        # Handle document listing errors
        logger.error(f"Document listing failed: {str(e)}")  # Log the error
        return jsonify({'error': str(e)}), 500  # Return error response

if __name__ == '__main__':
    """Run the Flask development server."""
    print("🚀 Starting RAG API Server...")  # Startup message
    print("📝 Upload documents and ask questions at: http://localhost:5000")  # Usage instructions
    print("⚙️  Change settings at: http://localhost:5000/settings")  # Settings page info
    
    app.run(debug=True, host='0.0.0.0', port=5000)  # Start development server 