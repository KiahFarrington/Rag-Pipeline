"""Sentence transformer embedder - Neural network based embeddings."""

import numpy as np  # Numerical computing for array operations
import logging  # Logging system for error tracking
import torch  # PyTorch for tensor operations and device management
from typing import List, Optional  # Type hints for better code clarity

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance for this embedder

# Global model to avoid reloading - store as None initially
_model = None  # Global variable to cache the SentenceTransformer model
_model_device = None  # Track which device the model is currently on


def _initialize_model() -> bool:
    """Initialize the SentenceTransformer model with proper device handling.
    
    Returns
    -------
    bool
        True if model initialized successfully, False otherwise
    """
    global _model, _model_device  # Access global model variables
    
    try:
        # Import here to avoid import errors if package not installed
        from sentence_transformers import SentenceTransformer  # Import SentenceTransformer library
        
        logger.info("Initializing SentenceTransformer model...")  # Log initialization start
        
        # Determine best available device for model
        if torch.cuda.is_available():
            device = 'cuda'  # Use GPU if available for faster processing
            logger.info("CUDA available - using GPU for embeddings")  # Log GPU usage
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Use Apple Silicon GPU if available
            logger.info("MPS available - using Apple Silicon GPU for embeddings")  # Log MPS usage
        else:
            device = 'cpu'  # Fall back to CPU processing
            logger.info("Using CPU for embeddings")  # Log CPU usage
        
        # Create model with explicit device handling to avoid meta tensor issues
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Initialize with device parameter
        _model_device = device  # Store device for reference
        
        # Verify model is properly loaded by doing a test encoding
        test_text = ["Test initialization"]  # Simple test text
        test_embedding = _model.encode(test_text)  # Test the model works
        
        logger.info(f"SentenceTransformer model initialized successfully on {device}")  # Log successful initialization
        logger.info(f"Model embedding dimension: {test_embedding.shape[1]}")  # Log embedding dimension
        
        return True  # Model initialized successfully
        
    except ImportError as e:
        logger.error("sentence-transformers package not installed. Run: pip install sentence-transformers")  # Log import error
        return False  # Failed to import required package
        
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformer model: {str(e)}")  # Log initialization error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        # Reset global variables on failure
        _model = None  # Clear failed model
        _model_device = None  # Clear device tracking
        return False  # Failed to initialize model


def create_sentence_transformer_embeddings(texts: List[str]) -> np.ndarray:
    """Create sentence embeddings using SentenceTransformers with proper error handling.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
        
    Returns
    -------
    np.ndarray
        Matrix of sentence embeddings (rows = texts, cols = 384 features)
        
    Raises
    ------
    ValueError
        If texts list is empty or contains invalid data
    RuntimeError
        If model initialization fails or encoding fails
    """
    global _model  # Access global model variable
    
    # Early validation of input parameters
    if not texts:
        logger.warning("Empty texts list provided to create_sentence_transformer_embeddings")  # Log empty input
        return np.array([])  # Return empty array for empty input
    
    # Validate that all texts are strings and not empty
    valid_texts = []  # List to store validated text strings
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            logger.warning(f"Text at index {i} is not a string: {type(text)}")  # Log type error
            continue  # Skip non-string items
        if not text.strip():
            logger.warning(f"Text at index {i} is empty or whitespace-only")  # Log empty text
            continue  # Skip empty texts
        valid_texts.append(text.strip())  # Add cleaned text to valid list
    
    # Check if we have any valid texts after filtering
    if not valid_texts:
        logger.error("No valid texts found after filtering")  # Log validation failure
        raise ValueError("No valid text strings provided for embedding")  # Raise error for no valid inputs
    
    logger.info(f"Processing {len(valid_texts)} valid texts for embedding")  # Log processing count
    
    # Initialize model if not already done
    if _model is None:
        logger.info("Model not initialized, attempting to initialize...")  # Log initialization attempt
        if not _initialize_model():
            logger.error("Failed to initialize SentenceTransformer model")  # Log initialization failure
            raise RuntimeError("Could not initialize SentenceTransformer model. Check logs for details.")  # Raise runtime error
    
    try:
        # Create embeddings for all texts using the initialized model
        logger.debug(f"Encoding {len(valid_texts)} texts...")  # Log encoding start
        embeddings = _model.encode(valid_texts)  # Generate embeddings using model
        
        # Validate output dimensions and data
        if embeddings is None or len(embeddings) == 0:
            logger.error("Model returned empty embeddings")  # Log empty result error
            raise RuntimeError("Model encoding returned empty result")  # Raise error for empty result
        
        # Convert to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)  # Convert to numpy array
        
        logger.info(f"Successfully created embeddings with shape: {embeddings.shape}")  # Log successful embedding creation
        
        return embeddings  # Return the generated embeddings
        
    except Exception as e:
        logger.error(f"Error during embedding creation: {str(e)}")  # Log embedding error
        logger.error(f"Error type: {type(e).__name__}")  # Log error type for debugging
        raise RuntimeError(f"Failed to create embeddings: {str(e)}")  # Raise runtime error with details


def create_single_sentence_transformer_embedding(text: str) -> np.ndarray:
    """Create sentence embedding for a single text with proper error handling.
    
    Perfect for vector databases and single queries.
    
    Parameters
    ----------
    text : str
        Single text string to embed
        
    Returns
    -------
    np.ndarray
        Single sentence embedding vector (384 dimensions for all-MiniLM-L6-v2)
        
    Raises
    ------
    ValueError
        If text is empty or invalid
    RuntimeError
        If embedding creation fails
    """
    # Early validation of input
    if not isinstance(text, str):
        logger.error(f"Input must be string, got {type(text)}")  # Log type error
        raise ValueError(f"Input must be string, got {type(text)}")  # Raise type error
    
    # Handle empty input
    if not text or not text.strip():
        logger.warning("Empty text provided to create_single_sentence_transformer_embedding")  # Log empty input warning
        return np.zeros(384)  # Return zero vector for empty input
    
    try:
        # Use batch function with single text for consistency
        embeddings = create_sentence_transformer_embeddings([text.strip()])  # Create embedding using batch function
        
        # Validate we got exactly one embedding
        if len(embeddings) != 1:
            logger.error(f"Expected 1 embedding, got {len(embeddings)}")  # Log unexpected result count
            raise RuntimeError(f"Expected 1 embedding, got {len(embeddings)}")  # Raise error for unexpected count
        
        # Return first (only) embedding
        return embeddings[0]  # Return the single embedding vector
        
    except Exception as e:
        logger.error(f"Error creating single embedding: {str(e)}")  # Log error details
        raise RuntimeError(f"Failed to create single embedding: {str(e)}")  # Re-raise with context


def get_sentence_dimension() -> int:
    """Get the dimension of sentence embeddings.
    
    Returns
    -------
    int
        Embedding dimension (384 for all-MiniLM-L6-v2)
    """
    return 384  # Return the fixed dimension for this model


def reset_model() -> None:
    """Reset the global model (useful for testing or memory cleanup).
    
    This function clears the cached model and forces reinitialization
    on the next embedding request.
    """
    global _model, _model_device  # Access global model variables
    
    logger.info("Resetting SentenceTransformer model cache")  # Log model reset
    _model = None  # Clear cached model
    _model_device = None  # Clear device tracking


def get_model_info() -> dict:
    """Get information about the current model state.
    
    Returns
    -------
    dict
        Dictionary containing model information including initialization status,
        device, and embedding dimension
    """
    global _model, _model_device  # Access global model variables
    
    return {
        'model_initialized': _model is not None,  # Whether model is loaded
        'model_device': _model_device,  # Device the model is on
        'embedding_dimension': get_sentence_dimension(),  # Embedding dimension
        'model_name': 'all-MiniLM-L6-v2'  # Model identifier
    } 