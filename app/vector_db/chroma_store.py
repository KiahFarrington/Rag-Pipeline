"""ChromaDB-based vector store implementation.

This module provides a vector database using ChromaDB, which is designed to be
developer-friendly with built-in persistence and metadata support.
"""

import numpy as np
import chromadb
from chromadb.config import Settings
import os
import uuid
from typing import List, Tuple, Dict, Any, Optional
from .base_vector_store import BaseVectorStore


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB-based vector store for semantic search.
    
    This implementation uses ChromaDB for vector storage and similarity search.
    ChromaDB is designed to be easy to use with automatic persistence and
    excellent metadata support.
    
    Features:
    - Automatic disk persistence (no manual save/load needed)
    - Built-in metadata filtering
    - Easy-to-use API
    - Cosine similarity search by default
    - Handles embedding normalization automatically
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_directory: Directory where ChromaDB will store data
        """
        # Store configuration
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True  # Allow resetting the database if needed
            )
        )
        
        # Get or create collection for storing documents
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Industrial protocol documentation embeddings"}
        )
        
        # Track embedding dimension (will be inferred from first batch)
        self._embedding_dim: Optional[int] = None
        
        print(f"ChromaDBVectorStore initialized with collection '{collection_name}' in '{persist_directory}'")
        print(f"Current collection size: {self._collection.count()} documents")

    def add_documents(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents with their embeddings to the ChromaDB vector store.
        
        Args:
            texts: List of document text chunks to store
            embeddings: Numpy array of shape (n_documents, embedding_dim) with document embeddings
            metadata: Optional list of metadata dictionaries for each document
            
        Returns:
            List of unique document IDs assigned to the added documents
        """
        # Validate input parameters
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy array")
        
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"Number of texts ({len(texts)}) must match number of embeddings ({embeddings.shape[0]})")
        
        # Set embedding dimension if not already set
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
            print(f"Inferred embedding dimension: {self._embedding_dim}")
        
        # Validate embedding dimensions match expected
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self._embedding_dim}")
        
        # Generate unique document IDs
        doc_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Prepare metadata (ChromaDB requires metadata for each document)
        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of texts ({len(texts)})")
        
        # Convert numpy array to list for ChromaDB
        embeddings_list = embeddings.astype(np.float32).tolist()
        
        # Add documents to ChromaDB collection
        # ChromaDB automatically handles persistence
        self._collection.add(
            ids=doc_ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadata
        )
        
        current_count = self._collection.count()
        print(f"Added {len(texts)} documents to ChromaDB. Total documents: {current_count}")
        return doc_ids

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
        """
        # Check if any documents have been added
        if self._collection.count() == 0:
            raise RuntimeError("No documents have been added to the vector store")
        
        # Validate query embedding dimensions
        if self._embedding_dim is not None and query_embedding.shape != (self._embedding_dim,):
            raise ValueError(f"Query embedding shape {query_embedding.shape} doesn't match expected ({self._embedding_dim},)")
        
        # Convert query embedding to list for ChromaDB
        query_embedding_list = query_embedding.astype(np.float32).tolist()
        
        # Prepare search parameters
        search_params = {
            "query_embeddings": [query_embedding_list],
            "n_results": min(top_k, self._collection.count())  # Don't search for more docs than we have
        }
        
        # Add metadata filter if provided (ChromaDB uses "where" parameter)
        if metadata_filter is not None:
            search_params["where"] = metadata_filter
        
        # Perform similarity search using ChromaDB
        results = self._collection.query(**search_params)
        
        # Extract results from ChromaDB response
        # ChromaDB returns results as lists of lists (for batch queries)
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        # Convert distances to similarity scores
        # ChromaDB returns squared euclidean distances, convert to similarity
        # For normalized embeddings, similarity = 1 - (distance^2 / 2)
        similarities = [1.0 - (dist / 2.0) for dist in distances]
        
        # Ensure we have metadata for each result (ChromaDB can return None)
        metadatas = [meta if meta is not None else {} for meta in metadatas]
        
        print(f"ChromaDB search returned {len(documents)} results out of {self._collection.count()} total documents")
        return documents, similarities, metadatas

    def get_document_count(self) -> int:
        """Get the total number of documents stored in the vector database."""
        return self._collection.count()
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings stored in this vector store."""
        return self._embedding_dim

    def save(self, filepath: str) -> None:
        """Save the ChromaDB vector store to disk for persistence.
        
        Note: ChromaDB automatically persists data, so this method just logs
        the current state. The actual data is always saved automatically.
        
        Args:
            filepath: Path identifier for logging (not used by ChromaDB)
        """
        doc_count = self._collection.count()
        print(f"ChromaDB vector store automatically persisted with {doc_count} documents")
        print(f"Data location: {self._persist_directory}")
        print(f"Collection: {self._collection_name}")
        
        # ChromaDB automatically persists, but we can force a manual persist if needed
        # This is mainly for demonstration - ChromaDB handles persistence automatically

    def load(self, filepath: str) -> None:
        """Load a previously saved ChromaDB vector store from disk.
        
        Note: ChromaDB automatically loads persisted data when the client is created,
        so this method just reconnects to the existing collection.
        
        Args:
            filepath: Path identifier (not used by ChromaDB - uses persist_directory from init)
        """
        # Reconnect to the collection (ChromaDB automatically loads persisted data)
        try:
            self._collection = self._client.get_collection(name=self._collection_name)
            doc_count = self._collection.count()
            print(f"Loaded ChromaDB collection '{self._collection_name}' with {doc_count} documents")
            
            # Try to infer embedding dimension from existing data
            if doc_count > 0 and self._embedding_dim is None:
                # Get one document to check embedding dimension
                sample = self._collection.get(limit=1, include=['embeddings'])
                if sample['embeddings'] and sample['embeddings'][0]:
                    self._embedding_dim = len(sample['embeddings'][0])
                    print(f"Inferred embedding dimension from existing data: {self._embedding_dim}")
                    
        except Exception as e:
            raise FileNotFoundError(f"Could not load ChromaDB collection '{self._collection_name}': {e}")

    def clear(self) -> None:
        """Remove all documents from the vector store."""
        # Delete the current collection
        self._client.delete_collection(name=self._collection_name)
        
        # Recreate an empty collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Industrial protocol documentation embeddings"}
        )
        
        # Reset embedding dimension
        self._embedding_dim = None
        
        print(f"Cleared all documents from ChromaDB collection '{self._collection_name}'") 