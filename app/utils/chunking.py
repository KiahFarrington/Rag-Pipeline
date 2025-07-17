"""Chunking Utilities

Provides unified interface for different text chunking strategies.
"""

import logging
from typing import List
from app.chunkers.semantic_chunker import chunk_by_semantics
from app.chunkers.fixed_length_chunker import chunk_by_fixed_length
from app.chunkers.adaptive_chunker import chunk_adaptively

logger = logging.getLogger(__name__)


def create_chunks_with_method(text: str, method: str, **config) -> List[str]:
    """Create chunks using the specified chunking method.
    
    Args:
        text: Text to chunk
        method: Chunking method ('fixed_length', 'semantic', 'adaptive')
        **config: Configuration parameters for chunking
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If method is not supported
    """
    # Get configuration with defaults
    chunk_size = config.get('chunk_size', 800)
    min_chunk_size = config.get('min_chunk_size', 100)
    max_chunk_size = config.get('max_chunk_size', 2000)
    
    logger.info(f"Chunking text using method: {method}")
    logger.debug(f"Text length: {len(text)} characters")
    
    try:
        # Apply chunking method based on configuration
        if method == 'semantic':
            chunks = chunk_by_semantics(text, min_chunk_size=min_chunk_size)
        elif method == 'adaptive':
            # Use adaptive chunking with configuration parameters
            chunks = chunk_adaptively(
                text,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                target_chunk_size=chunk_size
            )
        elif method == 'fixed_length':
            chunks = chunk_by_fixed_length(text, chunk_size=chunk_size)
        else:
            # Default to fixed_length for unknown methods
            logger.warning(f"Unknown chunking method '{method}', defaulting to fixed_length")
            chunks = chunk_by_fixed_length(text, chunk_size=chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks using {method} method")
        
        # Log chunk statistics
        if chunks:
            chunk_lengths = [len(chunk) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.debug(f"Average chunk length: {avg_length:.1f} characters")
            logger.debug(f"Chunk length range: {min(chunk_lengths)} - {max(chunk_lengths)}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Chunking failed with method {method}: {str(e)}")
        # Fallback to simple fixed-length chunking
        logger.info("Falling back to fixed-length chunking")
        return chunk_by_fixed_length(text, chunk_size=500)


def get_available_methods() -> List[str]:
    """Get list of available chunking methods.
    
    Returns:
        List of available chunking method names
    """
    return ['fixed_length', 'semantic', 'adaptive']


def validate_chunking_config(config: dict) -> bool:
    """Validate chunking configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check method is valid
        method = config.get('chunking_method')
        if method and method not in get_available_methods():
            logger.error(f"Invalid chunking method: {method}")
            return False
        
        # Check numeric parameters
        chunk_size = config.get('chunk_size', 800)
        min_chunk_size = config.get('min_chunk_size', 100)
        max_chunk_size = config.get('max_chunk_size', 2000)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.error(f"Invalid chunk_size: {chunk_size}")
            return False
            
        if not isinstance(min_chunk_size, int) or min_chunk_size <= 0:
            logger.error(f"Invalid min_chunk_size: {min_chunk_size}")
            return False
            
        if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
            logger.error(f"Invalid max_chunk_size: {max_chunk_size}")
            return False
        
        # Check logical relationships
        if min_chunk_size >= max_chunk_size:
            logger.error(f"min_chunk_size ({min_chunk_size}) must be less than max_chunk_size ({max_chunk_size})")
            return False
            
        if chunk_size < min_chunk_size or chunk_size > max_chunk_size:
            logger.error(f"chunk_size ({chunk_size}) must be between min_chunk_size ({min_chunk_size}) and max_chunk_size ({max_chunk_size})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating chunking config: {str(e)}")
        return False


def get_chunking_stats(chunks: List[str]) -> dict:
    """Get statistics about a list of chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary containing chunk statistics
    """
    if not chunks:
        return {
            'count': 0,
            'total_length': 0,
            'average_length': 0,
            'min_length': 0,
            'max_length': 0
        }
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    return {
        'count': len(chunks),
        'total_length': sum(chunk_lengths),
        'average_length': sum(chunk_lengths) / len(chunk_lengths),
        'min_length': min(chunk_lengths),
        'max_length': max(chunk_lengths)
    } 