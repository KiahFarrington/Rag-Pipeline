"""TF-IDF embedder - Simple word frequency based embeddings."""

import numpy as np  # Numerical computing for array operations
import logging  # Logging system for error tracking
from typing import List, Optional  # Type hints for better code clarity

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance for this embedder

# Global vectorizer to maintain consistency across calls
_vectorizer = None  # Global variable to cache the TF-IDF vectorizer


def create_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Create TF-IDF embeddings for a list of texts with proper error handling.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
        
    Returns
    -------
    np.ndarray
        Matrix of TF-IDF embeddings (rows = texts, cols = features)
        
    Raises
    ------
    ValueError
        If texts list is empty or contains invalid data
    RuntimeError
        If vectorizer creation or fitting fails
    """
    global _vectorizer  # Access global vectorizer variable
    
    # Early validation of input parameters
    if not texts:
        logger.warning("Empty texts list provided to create_tfidf_embeddings")  # Log empty input
        return np.array([]).reshape(0, 1000)  # Return empty array with correct shape for 1000-dimensional embeddings
    
    # Validate that all texts are strings and not empty
    valid_texts = []  # List to store validated text strings
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            logger.warning(f"Text at index {i} is not a string: {type(text)}")  # Log type error
            continue  # Skip non-string items
        if not text.strip():
            logger.warning(f"Text at index {i} is empty or whitespace-only")  # Log empty text
            valid_texts.append("empty text")  # Use placeholder for empty texts to maintain array shape
        else:
            valid_texts.append(text.strip())  # Add cleaned text to valid list
    
    # Check if we have any valid texts after filtering
    if not valid_texts:
        logger.error("No valid texts found after filtering")  # Log validation failure
        return np.zeros((len(texts), 1000))  # Return zero embeddings to maintain expected shape
    
    logger.info(f"Processing {len(valid_texts)} valid texts for TF-IDF embedding")  # Log processing count
    
    try:
        # Import scikit-learn here to handle import errors gracefully
        from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer
        
        # Create TF-IDF vectorizer with robust configuration
        # max_features limits vocabulary size for memory efficiency and speed
        # stop_words removes common English words like 'the', 'and', 'is'
        # lowercase normalizes text for consistency
        # min_df=1 ensures we include words that appear at least once
        # max_df=0.95 excludes words that appear in >95% of documents (too common)
        _vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary to most important 1000 words
            stop_words='english',  # Remove common English stop words
            lowercase=True,  # Convert all text to lowercase for consistency
            min_df=1,  # Include words that appear at least once
            max_df=1.0,  # Include all words (fixed for small document sets)
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only include alphabetic tokens with 2+ characters
        )
        
        logger.debug("TF-IDF vectorizer created successfully")  # Log vectorizer creation
        
        # Fit and transform texts to TF-IDF vectors
        logger.debug(f"Fitting vectorizer on {len(valid_texts)} texts...")  # Log fitting start
        tfidf_matrix = _vectorizer.fit_transform(valid_texts)  # Train vectorizer and create embeddings
        
        # Validate the output matrix
        if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
            logger.error("TF-IDF vectorizer returned empty matrix")  # Log empty result error
            raise RuntimeError("TF-IDF vectorization failed - empty result")  # Raise error for empty result
        
        # Convert sparse matrix to dense numpy array for consistency
        dense_matrix = tfidf_matrix.toarray()  # Convert to dense format
        
        logger.info(f"Successfully created TF-IDF embeddings with shape: {dense_matrix.shape}")  # Log success
        logger.info(f"Vocabulary size: {len(_vectorizer.vocabulary_)}")  # Log vocabulary size
        
        return dense_matrix  # Return the TF-IDF embedding matrix
        
    except ImportError as e:
        logger.error("scikit-learn package not installed. Run: pip install scikit-learn")  # Log import error
        raise RuntimeError("scikit-learn package required for TF-IDF embeddings")  # Raise runtime error
        
    except Exception as e:
        logger.error(f"Error during TF-IDF embedding creation: {str(e)}")  # Log embedding error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        # Reset global vectorizer on failure
        _vectorizer = None  # Clear failed vectorizer
        raise RuntimeError(f"Failed to create TF-IDF embeddings: {str(e)}")  # Raise runtime error with details


def create_single_tfidf_embedding(text: str) -> np.ndarray:
    """Create TF-IDF embedding for a single text with proper error handling.
    
    This function is perfect for vector databases and single queries.
    Note: Must call create_tfidf_embeddings first to train the vectorizer.
    
    Parameters
    ----------
    text : str
        Single text string to embed
        
    Returns
    -------
    np.ndarray
        Single TF-IDF embedding vector (1000 dimensions)
        
    Raises
    ------
    ValueError
        If text is empty or invalid
    RuntimeError
        If vectorizer is not trained or transformation fails
    """
    global _vectorizer  # Access global vectorizer variable
    
    # Early validation of input
    if not isinstance(text, str):
        logger.error(f"Input must be string, got {type(text)}")  # Log type error
        raise ValueError(f"Input must be string, got {type(text)}")  # Raise type error
    
    # Handle empty input
    if not text or not text.strip():
        logger.warning("Empty text provided to create_single_tfidf_embedding")  # Log empty input warning
        return np.zeros(1000)  # Return zero vector for empty input
    
    # Clean the input text
    clean_text = text.strip()  # Remove leading/trailing whitespace
    
    # Check if vectorizer is trained
    if _vectorizer is None:
        logger.warning("No trained vectorizer found, creating new one for single text")  # Log missing vectorizer
        
        try:
            # Import scikit-learn here to handle import errors gracefully
            from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer
            
            # If no vectorizer, create one with just this text
            # This is less ideal but works for single queries
            _vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit vocabulary to 1000 words
                stop_words='english',  # Remove common English stop words
                lowercase=True,  # Convert to lowercase for consistency
                min_df=1,  # Include words that appear at least once
                max_df=1.0,  # Include all words (consistent with main config)
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens with 2+ characters
            )
            _vectorizer.fit([clean_text])  # Train vectorizer on single text
            
            logger.info("Created new TF-IDF vectorizer for single text")  # Log vectorizer creation
            
        except ImportError as e:
            logger.error("scikit-learn package not installed. Run: pip install scikit-learn")  # Log import error
            raise RuntimeError("scikit-learn package required for TF-IDF embeddings")  # Raise runtime error
            
        except Exception as e:
            logger.error(f"Failed to create vectorizer for single text: {str(e)}")  # Log creation error
            raise RuntimeError(f"Failed to create TF-IDF vectorizer: {str(e)}")  # Raise runtime error
    
    try:
        # Transform single text to vector using trained vectorizer
        logger.debug(f"Transforming single text to TF-IDF vector...")  # Log transformation start
        vector = _vectorizer.transform([clean_text])  # Transform text to TF-IDF vector
        
        # Validate the output vector
        if vector is None or vector.shape[0] == 0:
            logger.error("TF-IDF transformation returned empty vector")  # Log empty result error
            raise RuntimeError("TF-IDF transformation failed - empty result")  # Raise error for empty result
        
        # Convert to dense format and extract single vector
        dense_vector = vector.toarray()[0]  # Convert to dense array and get first (only) vector
        
        logger.debug(f"Successfully created single TF-IDF embedding with {np.count_nonzero(dense_vector)} non-zero features")  # Log success
        
        # Return as 1D array (single vector)
        return dense_vector  # Return the single TF-IDF embedding vector
        
    except Exception as e:
        logger.error(f"Error transforming single text to TF-IDF: {str(e)}")  # Log transformation error
        raise RuntimeError(f"Failed to create single TF-IDF embedding: {str(e)}")  # Raise runtime error with details


def get_tfidf_dimension() -> int:
    """Get the dimension of TF-IDF embeddings.
    
    Returns
    -------
    int
        Maximum embedding dimension (1000 features)
    """
    return 1000  # Return the fixed dimension for TF-IDF embeddings


def reset_tfidf_vectorizer() -> None:
    """Reset the global vectorizer (useful for testing or memory cleanup).
    
    This function clears the cached vectorizer and forces retraining
    on the next embedding request.
    """
    global _vectorizer  # Access global vectorizer variable
    
    logger.info("Resetting TF-IDF vectorizer cache")  # Log vectorizer reset
    _vectorizer = None  # Clear cached vectorizer


def get_vectorizer_info() -> dict:
    """Get information about the current vectorizer state.
    
    Returns
    -------
    dict
        Dictionary containing vectorizer information including training status,
        vocabulary size, and embedding dimension
    """
    global _vectorizer  # Access global vectorizer variable
    
    # Check if vectorizer is trained and get vocabulary size
    vocab_size = 0  # Default vocabulary size
    if _vectorizer is not None and hasattr(_vectorizer, 'vocabulary_'):
        vocab_size = len(_vectorizer.vocabulary_)  # Get actual vocabulary size
    
    return {
        'vectorizer_trained': _vectorizer is not None,  # Whether vectorizer is trained
        'vocabulary_size': vocab_size,  # Size of learned vocabulary
        'max_features': 1000,  # Maximum number of features
        'embedding_dimension': get_tfidf_dimension(),  # Embedding dimension
        'method': 'TF-IDF'  # Embedding method identifier
    } 