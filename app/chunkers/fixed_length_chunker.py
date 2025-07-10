"""Fixed-length text chunker - Simple implementation with robust error handling."""

import logging  # Logging system for error tracking
from typing import List  # Type hints for better code clarity

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance for this chunker


def chunk_by_fixed_length(text: str, chunk_size: int = 2000) -> List[str]:
    """Split text into fixed-length chunks with proper error handling and validation.
    
    Parameters
    ----------
    text : str
        Input text to be chunked
    chunk_size : int, optional
        Size of each chunk in characters (default: 500)
        
    Returns
    -------
    List[str]
        List of text chunks, each approximately chunk_size characters
        
    Raises
    ------
    ValueError
        If text is not a string or chunk_size is invalid
    RuntimeError
        If chunking process fails
    """
    # Early validation of input parameters
    if not isinstance(text, str):
        logger.error(f"Text must be string, got {type(text)}")  # Log type error
        raise ValueError(f"Text must be string, got {type(text)}")  # Raise type error
    
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.error(f"Chunk size must be positive integer, got {chunk_size}")  # Log invalid chunk size
        raise ValueError(f"Chunk size must be positive integer, got {chunk_size}")  # Raise value error
    
    # Handle empty input gracefully
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided to chunk_by_fixed_length")  # Log empty input
        return []  # Return empty list for empty input
    
    # Clean the input text by removing leading/trailing whitespace
    clean_text = text.strip()  # Remove leading and trailing whitespace
    
    # Log the chunking process start
    logger.info(f"Chunking text of {len(clean_text)} characters into {chunk_size}-character chunks")  # Log process start
    
    try:
        # Split text into fixed-length chunks
        chunks = []  # List to store text chunks
        
        # Iterate through text in chunk_size increments
        for i in range(0, len(clean_text), chunk_size):
            # Extract chunk from current position to current position + chunk_size
            chunk = clean_text[i:i + chunk_size]  # Extract substring for this chunk
            
            # Validate chunk before adding
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)  # Add chunk to results list
                logger.debug(f"Created chunk {len(chunks)} with {len(chunk)} characters")  # Log chunk creation
        
        # Validate results before returning
        if not chunks:
            logger.warning("No chunks created from input text")  # Log no chunks warning
            return []  # Return empty list if no chunks created
        
        # Log successful chunking completion
        logger.info(f"Successfully created {len(chunks)} chunks from text")  # Log completion
        
        # Return the chunks list
        return chunks  # Return list of text chunks
        
    except Exception as e:
        # Handle any unexpected errors during chunking
        logger.error(f"Error during fixed-length chunking: {str(e)}")  # Log chunking error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        raise RuntimeError(f"Fixed-length chunking failed: {str(e)}")  # Raise runtime error with details


def get_optimal_chunk_size(text_length: int, target_chunks: int = 10) -> int:
    """Calculate optimal chunk size based on text length and target number of chunks.
    
    Parameters
    ----------
    text_length : int
        Length of the text to be chunked
    target_chunks : int, optional
        Desired number of chunks (default: 10)
        
    Returns
    -------
    int
        Recommended chunk size in characters
        
    Raises
    ------
    ValueError
        If text_length or target_chunks are invalid
    """
    # Validate input parameters
    if not isinstance(text_length, int) or text_length <= 0:
        logger.error(f"Text length must be positive integer, got {text_length}")  # Log invalid text length
        raise ValueError(f"Text length must be positive integer, got {text_length}")  # Raise value error
    
    if not isinstance(target_chunks, int) or target_chunks <= 0:
        logger.error(f"Target chunks must be positive integer, got {target_chunks}")  # Log invalid target chunks
        raise ValueError(f"Target chunks must be positive integer, got {target_chunks}")  # Raise value error
    
    # Calculate optimal chunk size
    optimal_size = max(100, text_length // target_chunks)  # Ensure minimum 100 characters per chunk
    
    logger.info(f"Calculated optimal chunk size: {optimal_size} for text of {text_length} characters")  # Log calculation
    
    return optimal_size  # Return calculated chunk size


def get_chunker_info() -> dict:
    """Get information about the fixed-length chunker.
    
    Returns
    -------
    dict
        Dictionary containing chunker information and configuration
    """
    return {
        'method': 'fixed_length',  # Chunking method identifier
        'default_chunk_size': 500,  # Default chunk size in characters
        'minimum_chunk_size': 1,  # Minimum allowed chunk size
        'supports_overlap': False,  # Whether this chunker supports overlapping chunks
        'description': 'Splits text into fixed-size character chunks'  # Method description
    } 