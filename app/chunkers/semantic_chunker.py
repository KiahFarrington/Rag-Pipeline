"""Semantic text chunker - Simple implementation with robust error handling."""

import re  # Regular expressions for text pattern matching
import logging  # Logging system for error tracking
from typing import List  # Type hints for better code clarity

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance for this chunker


def chunk_by_semantics(text: str, min_chunk_size: int = 50) -> List[str]:
    """Split text based on semantic boundaries (paragraphs) with proper error handling.
    
    Parameters
    ----------
    text : str
        Input text to be chunked
    min_chunk_size : int, optional
        Minimum size for chunks in characters (default: 50)
        
    Returns
    -------
    List[str]
        List of text chunks split by paragraph breaks
        
    Raises
    ------
    ValueError
        If text is not a string or min_chunk_size is invalid
    RuntimeError
        If chunking process fails
    """
    # Early validation of input parameters
    if not isinstance(text, str):
        logger.error(f"Text must be string, got {type(text)}")  # Log type error
        raise ValueError(f"Text must be string, got {type(text)}")  # Raise type error
    
    if not isinstance(min_chunk_size, int) or min_chunk_size < 0:
        logger.error(f"Minimum chunk size must be non-negative integer, got {min_chunk_size}")  # Log invalid chunk size
        raise ValueError(f"Minimum chunk size must be non-negative integer, got {min_chunk_size}")  # Raise value error
    
    # Handle empty input gracefully
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided to chunk_by_semantics")  # Log empty input
        return []  # Return empty list for empty input
    
    # Clean the input text by removing leading/trailing whitespace
    clean_text = text.strip()  # Remove leading and trailing whitespace
    
    # Log the chunking process start
    logger.info(f"Performing semantic chunking on text of {len(clean_text)} characters")  # Log process start
    
    try:
        # Split by paragraph breaks using robust pattern
        # Matches double newlines with optional spaces or newlines with 3+ spaces
        # This pattern handles various paragraph break formats
        paragraph_pattern = r'\n\s*\n|\n\s{3,}'  # Regex pattern for paragraph breaks
        
        logger.debug(f"Using paragraph pattern: {paragraph_pattern}")  # Log pattern used
        
        # Split text using the paragraph pattern
        raw_chunks = re.split(paragraph_pattern, clean_text)  # Split text by paragraph breaks
        
        logger.debug(f"Initial split produced {len(raw_chunks)} raw chunks")  # Log initial split count
        
        # Clean up chunks and remove empty ones
        chunks = []  # List to store validated text chunks
        
        for i, chunk in enumerate(raw_chunks):
            # Clean each chunk by stripping whitespace
            cleaned_chunk = chunk.strip()  # Remove leading/trailing whitespace from chunk
            
            # Only add chunks that meet minimum size requirement
            if cleaned_chunk and len(cleaned_chunk) >= min_chunk_size:
                chunks.append(cleaned_chunk)  # Add valid chunk to results
                logger.debug(f"Added chunk {len(chunks)} with {len(cleaned_chunk)} characters")  # Log chunk addition
            elif cleaned_chunk:
                logger.debug(f"Skipped chunk {i+1} (too small: {len(cleaned_chunk)} < {min_chunk_size} chars)")  # Log skipped chunk
        
        # If no paragraph breaks found or no valid chunks, handle as single chunk
        if not chunks:
            logger.warning("No valid semantic chunks found, treating as single chunk")  # Log fallback to single chunk
            
            # Check if original text meets minimum size requirement
            if len(clean_text) >= min_chunk_size:
                chunks = [clean_text]  # Return whole text as single chunk
                logger.info("Using entire text as single chunk")  # Log single chunk usage
            else:
                logger.warning(f"Text too small ({len(clean_text)} < {min_chunk_size} chars), returning empty list")  # Log text too small
                return []  # Return empty list for text that's too small
        
        # Log successful chunking completion
        logger.info(f"Successfully created {len(chunks)} semantic chunks")  # Log completion
        
        # Return the chunks list
        return chunks  # Return list of semantically chunked text
        
    except re.error as e:
        # Handle regular expression errors
        logger.error(f"Regular expression error during semantic chunking: {str(e)}")  # Log regex error
        raise RuntimeError(f"Semantic chunking failed due to regex error: {str(e)}")  # Raise runtime error
        
    except Exception as e:
        # Handle any other unexpected errors during chunking
        logger.error(f"Error during semantic chunking: {str(e)}")  # Log chunking error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        raise RuntimeError(f"Semantic chunking failed: {str(e)}")  # Raise runtime error with details


def detect_text_structure(text: str) -> dict:
    """Analyze text structure to provide insights for semantic chunking.
    
    Parameters
    ----------
    text : str
        Input text to analyze
        
    Returns
    -------
    dict
        Dictionary containing text structure analysis
        
    Raises
    ------
    ValueError
        If text is not a string
    """
    # Validate input parameter
    if not isinstance(text, str):
        logger.error(f"Text must be string, got {type(text)}")  # Log type error
        raise ValueError(f"Text must be string, got {type(text)}")  # Raise type error
    
    # Handle empty input
    if not text or not text.strip():
        logger.warning("Empty text provided to detect_text_structure")  # Log empty input
        return {
            'total_length': 0,  # Total character count
            'paragraph_breaks': 0,  # Number of paragraph breaks found
            'line_breaks': 0,  # Number of line breaks found
            'sentences': 0,  # Estimated number of sentences
            'words': 0  # Estimated number of words
        }
    
    # Clean the input text
    clean_text = text.strip()  # Remove leading/trailing whitespace
    
    try:
        # Count various text structure elements
        paragraph_breaks = len(re.findall(r'\n\s*\n|\n\s{3,}', clean_text))  # Count paragraph breaks
        line_breaks = clean_text.count('\n')  # Count all line breaks
        sentences = len(re.findall(r'[.!?]+', clean_text))  # Count sentence-ending punctuation
        words = len(clean_text.split())  # Count words by splitting on whitespace
        
        # Log analysis results
        logger.debug(f"Text structure analysis: {len(clean_text)} chars, {words} words, {sentences} sentences")  # Log analysis
        
        return {
            'total_length': len(clean_text),  # Total character count
            'paragraph_breaks': paragraph_breaks,  # Number of paragraph breaks found
            'line_breaks': line_breaks,  # Number of line breaks found
            'sentences': sentences,  # Estimated number of sentences
            'words': words  # Estimated number of words
        }
        
    except Exception as e:
        # Handle analysis errors
        logger.error(f"Error analyzing text structure: {str(e)}")  # Log analysis error
        raise RuntimeError(f"Text structure analysis failed: {str(e)}")  # Raise runtime error


def get_chunker_info() -> dict:
    """Get information about the semantic chunker.
    
    Returns
    -------
    dict
        Dictionary containing chunker information and configuration
    """
    return {
        'method': 'semantic',  # Chunking method identifier
        'default_min_chunk_size': 50,  # Default minimum chunk size in characters
        'pattern_type': 'paragraph_breaks',  # Type of pattern used for splitting
        'supports_structure_analysis': True,  # Whether this chunker supports text structure analysis
        'description': 'Splits text at paragraph boundaries for semantic coherence'  # Method description
    } 