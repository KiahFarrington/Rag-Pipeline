"""Semantic text chunker - Simple implementation."""

import re

def chunk_by_semantics(text: str) -> list[str]:
    """Split text based on semantic boundaries (paragraphs).
    
    Parameters
    ----------
    text : str
        Input text to be chunked
        
    Returns
    -------
    list[str]
        List of text chunks split by paragraph breaks
    """
    # Handle empty input
    if not text or not text.strip():
        return []
    
    # Clean the input text
    text = text.strip()
    
    # Split by paragraph breaks using robust pattern
    # Matches double newlines with optional spaces or newlines with 3+ spaces
    paragraph_pattern = r'\n\s*\n|\n\s{3,}'
    chunks = re.split(paragraph_pattern, text)
    
    # Clean up chunks and remove empty ones
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # If no paragraph breaks found, return whole text as single chunk
    if not chunks:
        return [text]
    
    # Return the chunks
    return chunks 