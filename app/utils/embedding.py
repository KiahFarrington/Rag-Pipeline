"""Embedding Utilities

Provides unified interface for different text embedding methods.
"""

import logging
import numpy as np
from typing import List, Union
from app.embedders.tfidf_embedder import create_tfidf_embeddings, create_single_tfidf_embedding
from app.embedders.sentence_transformer_embedder import create_sentence_transformer_embeddings, create_single_sentence_transformer_embedding

logger = logging.getLogger(__name__)


def create_embeddings_with_method(chunks: List[str], method: str) -> List[np.ndarray]:
    """Create embeddings using the specified embedding method.
    
    Args:
        chunks: List of text chunks to embed
        method: Embedding method ('tfidf', 'sentence_transformer')
        
    Returns:
        List of embedding arrays
        
    Raises:
        ValueError: If method is not supported
    """
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return []
    
    logger.info(f"Creating embeddings for {len(chunks)} chunks using method: {method}")
    
    try:
        if method == 'tfidf':
            embeddings = create_tfidf_embeddings(chunks)
        elif method == 'sentence_transformer':
            embeddings = create_sentence_transformer_embeddings(chunks)
        else:
            # Default to TF-IDF for unknown methods
            logger.warning(f"Unknown embedding method '{method}', defaulting to tfidf")
            embeddings = create_tfidf_embeddings(chunks)
        
        logger.info(f"Created {len(embeddings)} embeddings using {method} method")
        
        # Log embedding statistics
        if embeddings is not None and len(embeddings) > 0:
            if hasattr(embeddings, 'shape'):
                logger.debug(f"Embedding shape: {embeddings.shape}")
            else:
                embedding_dims = [emb.shape[0] if hasattr(emb, 'shape') else len(emb) for emb in embeddings]
                logger.debug(f"Embedding dimensions: {embedding_dims[0] if embedding_dims else 'unknown'}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding creation failed with method {method}: {str(e)}")
        # Fallback to TF-IDF
        logger.info("Falling back to TF-IDF embeddings")
        return create_tfidf_embeddings(chunks)


def create_query_embedding_with_method(query: str, method: str) -> Union[np.ndarray, None]:
    """Create embedding for a single query using the specified method.
    
    Args:
        query: Query text to embed
        method: Embedding method ('tfidf', 'sentence_transformer')
        
    Returns:
        Query embedding array or None if failed
    """
    if not query.strip():
        logger.warning("Empty query provided for embedding")
        return None
    
    logger.debug(f"Creating query embedding using method: {method}")
    
    try:
        if method == 'tfidf':
            embedding = create_single_tfidf_embedding(query)
        elif method == 'sentence_transformer':
            embedding = create_single_sentence_transformer_embedding(query)
        else:
            # Default to TF-IDF for unknown methods
            logger.warning(f"Unknown embedding method '{method}', defaulting to tfidf")
            embedding = create_single_tfidf_embedding(query)
        
        logger.debug(f"Created query embedding with {method} method")
        return embedding
        
    except Exception as e:
        logger.error(f"Query embedding creation failed with method {method}: {str(e)}")
        # Fallback to TF-IDF
        logger.info("Falling back to TF-IDF for query embedding")
        try:
            return create_single_tfidf_embedding(query)
        except Exception as fallback_error:
            logger.error(f"Fallback embedding also failed: {str(fallback_error)}")
            return None


def get_available_methods() -> List[str]:
    """Get list of available embedding methods.
    
    Returns:
        List of available embedding method names
    """
    return ['tfidf', 'sentence_transformer']


def validate_embedding_config(config: dict) -> bool:
    """Validate embedding configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check method is valid
        method = config.get('embedding_method')
        if method and method not in get_available_methods():
            logger.error(f"Invalid embedding method: {method}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating embedding config: {str(e)}")
        return False


def get_embedding_info(method: str) -> dict:
    """Get information about an embedding method.
    
    Args:
        method: Embedding method name
        
    Returns:
        Dictionary containing method information
    """
    method_info = {
        'tfidf': {
            'name': 'TF-IDF',
            'description': 'Term Frequency-Inverse Document Frequency',
            'dependencies': ['scikit-learn'],
            'local': True,
            'speed': 'fast',
            'quality': 'good for keyword matching'
        },
        'sentence_transformer': {
            'name': 'Sentence Transformer',
            'description': 'Neural sentence embeddings',
            'dependencies': ['sentence-transformers', 'torch'],
            'local': True,
            'speed': 'medium',
            'quality': 'excellent for semantic similarity'
        }
    }
    
    return method_info.get(method, {
        'name': 'Unknown',
        'description': 'Unknown embedding method',
        'dependencies': [],
        'local': False,
        'speed': 'unknown',
        'quality': 'unknown'
    })


def test_embedding_method(method: str, test_text: str = "This is a test sentence.") -> dict:
    """Test an embedding method with sample text.
    
    Args:
        method: Embedding method to test
        test_text: Text to use for testing
        
    Returns:
        Dictionary containing test results
    """
    try:
        import time
        
        logger.info(f"Testing embedding method: {method}")
        
        start_time = time.time()
        embedding = create_query_embedding_with_method(test_text, method)
        end_time = time.time()
        
        if embedding is not None:
            return {
                'status': 'success',
                'method': method,
                'embedding_dim': embedding.shape[0] if hasattr(embedding, 'shape') else len(embedding),
                'processing_time': end_time - start_time,
                'test_text': test_text
            }
        else:
            return {
                'status': 'failed',
                'method': method,
                'error': 'Embedding creation returned None'
            }
            
    except Exception as e:
        logger.error(f"Embedding test failed for method {method}: {str(e)}")
        return {
            'status': 'error',
            'method': method,
            'error': str(e)
        }


def compare_embedding_methods(test_text: str = "This is a test sentence.") -> dict:
    """Compare all available embedding methods.
    
    Args:
        test_text: Text to use for comparison
        
    Returns:
        Dictionary containing comparison results
    """
    results = {}
    
    for method in get_available_methods():
        results[method] = test_embedding_method(method, test_text)
    
    return {
        'test_text': test_text,
        'methods': results,
        'available_methods': get_available_methods()
    } 