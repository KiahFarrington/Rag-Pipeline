"""Abstract base class for vector database implementations.

This module defines the common interface that all vector store implementations
must follow, enabling easy switching between different vector databases.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BaseVectorStore(ABC):
    """Abstract base class for vector database implementations.
    
    This class defines the common interface that all vector stores must implement,
    whether they are local (FAISS, ChromaDB) or cloud-based (Pinecone, Weaviate).
    
    The interface supports:
    - Adding documents with embeddings and metadata
    - Similarity search with configurable results count
    - Persistence operations for saving/loading
    - Metadata filtering for complex queries
    """

    @abstractmethod
    def add_documents(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents with their embeddings to the vector store.
        
        Args:
            texts: List of document text chunks to store
            embeddings: Numpy array of shape (n_documents, embedding_dim) containing document embeddings
            metadata: Optional list of metadata dictionaries for each document (document source, page numbers, etc.)
            
        Returns:
            List of unique document IDs assigned to the added documents
            
        Raises:
            ValueError: If texts and embeddings have different lengths
            TypeError: If embeddings is not a numpy array
        """
        pass

    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Search for documents most similar to the query embedding.
        
        Args:
            query_embedding: Numpy array of shape (embedding_dim,) representing the query
            top_k: Maximum number of similar documents to return
            metadata_filter: Optional dictionary to filter results by metadata fields
            
        Returns:
            Tuple containing:
            - List of document texts (most similar first)
            - List of similarity scores (higher = more similar)  
            - List of metadata dictionaries for each result
            
        Raises:
            ValueError: If query_embedding has wrong dimensions
            RuntimeError: If no documents have been added to the store
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents stored in the vector database.
        
        Returns:
            Integer count of documents currently stored
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the vector store to disk for persistence.
        
        Args:
            filepath: Path where the vector store should be saved
            
        Raises:
            IOError: If the file cannot be written to the specified path
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load a previously saved vector store from disk.
        
        Args:
            filepath: Path to the saved vector store file
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            IOError: If the file cannot be read or is corrupted
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the vector store.
        
        This operation cannot be undone. Use save() before calling if you want
        to preserve the current state.
        """
        pass

    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings stored in this vector store.
        
        Returns:
            Integer dimension of embeddings, or None if no documents stored yet
        """
        # Default implementation - can be overridden by subclasses
        return None 