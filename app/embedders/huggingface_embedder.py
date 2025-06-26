"""HuggingFace embedder - Free transformer based embeddings."""

import numpy as np

# Global model components to avoid reloading
_tokenizer = None
_model = None

def create_huggingface_embeddings(texts: list[str]) -> np.ndarray:
    """Create embeddings using free HuggingFace transformers.
    
    Parameters
    ----------
    texts : list[str]
        List of text strings to embed
        
    Returns
    -------
    np.ndarray
        Matrix of transformer embeddings (rows = texts, cols = 768 features)
    """
    global _tokenizer, _model
    
    # Handle empty input
    if not texts:
        return np.array([])
    
    # Import here to avoid requiring dependency at module load time
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Load free BERT model once (distilbert is smaller and faster)
    if _tokenizer is None or _model is None:
        model_name = "distilbert-base-uncased"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        
        # Set model to evaluation mode
        _model.eval()
    
    embeddings = []
    
    # Process each text
    for text in texts:
        # Tokenize text
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get embeddings without computing gradients
        with torch.no_grad():
            outputs = _model(**inputs)
            
            # Use mean pooling of last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(embedding.numpy())
    
    return np.array(embeddings)


def create_single_huggingface_embedding(text: str) -> np.ndarray:
    """Create embedding for a single text using HuggingFace transformers.
    
    Perfect for vector databases and single queries.
    
    Parameters
    ----------
    text : str
        Single text string to embed
        
    Returns
    -------
    np.ndarray
        Single transformer embedding vector
    """
    # Handle empty input
    if not text or not text.strip():
        return np.zeros(768)
    
    # Use batch function with single text
    embeddings = create_huggingface_embeddings([text])
    
    # Return first (only) embedding
    return embeddings[0] if len(embeddings) > 0 else np.zeros(768)


def get_huggingface_dimension() -> int:
    """Get the dimension of HuggingFace embeddings.
    
    Returns
    -------
    int
        Embedding dimension (768 for distilbert-base-uncased)
    """
    return 768 