"""Sentence transformer embedder - Neural network based embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model to avoid reloading
_model = None


def create_sentence_transformer_embeddings(texts: list[str]) -> np.ndarray:
    """Create sentence embeddings using SentenceTransformers.
    
    Parameters
    ----------
    texts : list[str]
        List of text strings to embed
        
    Returns
    -------
    np.ndarray
        Matrix of sentence embeddings (rows = texts, cols = 384 features)
    """
    global _model
    
    # Handle empty input
    if not texts:
        return np.array([])
    
    # Load pre-trained model once (small and fast)
    # all-MiniLM-L6-v2 is lightweight but good quality
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for all texts
    embeddings = _model.encode(texts)
    
    return embeddings


def create_single_sentence_transformer_embedding(text: str) -> np.ndarray:
    """Create sentence embedding for a single text.
    
    Perfect for vector databases and single queries.
    
    Parameters
    ----------
    text : str
        Single text string to embed
        
    Returns
    -------
    np.ndarray
        Single sentence embedding vector
    """
    # Handle empty input
    if not text or not text.strip():
        return np.zeros(384)
    
    # Use batch function with single text
    embeddings = create_sentence_transformer_embeddings([text])
    
    # Return first (only) embedding
    return embeddings[0] if len(embeddings) > 0 else np.zeros(384)


def get_sentence_dimension() -> int:
    """Get the dimension of sentence embeddings.
    
    Returns
    -------
    int
        Embedding dimension (384 for all-MiniLM-L6-v2)
    """
    return 384 