"""TF-IDF embedder - Simple word frequency based embeddings."""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Global vectorizer to maintain consistency across calls
_vectorizer = None

def create_tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Create TF-IDF embeddings for a list of texts.
    
    Parameters
    ----------
    texts : list[str]
        List of text strings to embed
        
    Returns
    -------
    np.ndarray
        Matrix of TF-IDF embeddings (rows = texts, cols = features)
    """
    global _vectorizer
    
    # Handle empty input
    if not texts:
        return np.array([])
    
    # Create TF-IDF vectorizer
    # max_features limits vocabulary size for simplicity
    # stop_words removes common words like 'the', 'and'
    _vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform texts to TF-IDF vectors
    tfidf_matrix = _vectorizer.fit_transform(texts)
    
    # Convert sparse matrix to dense numpy array
    return tfidf_matrix.toarray()


def create_single_tfidf_embedding(text: str) -> np.ndarray:
    """Create TF-IDF embedding for a single text.
    
    This function is perfect for vector databases and single queries.
    Note: Must call create_tfidf_embeddings first to train the vectorizer.
    
    Parameters
    ----------
    text : str
        Single text string to embed
        
    Returns
    -------
    np.ndarray
        Single TF-IDF embedding vector
    """
    global _vectorizer
    
    # Handle empty input
    if not text or not text.strip():
        return np.zeros(1000)  # Return zero vector
    
    # Check if vectorizer is trained
    if _vectorizer is None:
        # If no vectorizer, create one with just this text
        # This is less ideal but works for single queries
        _vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )
        _vectorizer.fit([text])
    
    # Transform single text to vector
    vector = _vectorizer.transform([text])
    
    # Return as 1D array (single vector)
    return vector.toarray()[0]


def get_tfidf_dimension() -> int:
    """Get the dimension of TF-IDF embeddings.
    
    Returns
    -------
    int
        Maximum embedding dimension (1000 features)
    """
    return 1000


def reset_tfidf_vectorizer():
    """Reset the global vectorizer (useful for testing)."""
    global _vectorizer
    _vectorizer = None 