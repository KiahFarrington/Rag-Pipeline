"""Fixed-length text chunker - Simple implementation."""

def chunk_by_fixed_length(text: str) -> list[str]:
    """Split text into fixed-length chunks.
    
    Parameters
    ----------
    text : str
        Input text to be chunked
        
    Returns
    -------
    list[str]
        List of text chunks
    """
    # Handle empty input
    if not text or not text.strip():
        return []
    
    # Clean the input text
    text = text.strip()
    
    # Split text into 500-character chunks
    chunks = []
    chunk_size = 500
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    
    # Return the chunks
    return chunks 