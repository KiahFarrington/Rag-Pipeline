"""FAISS-based vector store implementation.

This module provides a high-performance vector database using Facebook's FAISS library
for similarity search. FAISS is optimized for fast nearest neighbor search on dense vectors.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Dict, Any, Optional
from .base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for efficient similarity search.
    
    This implementation uses Facebook's FAISS library for fast approximate 
    nearest neighbor search. It stores embeddings in memory and provides
    high-performance similarity search capabilities.
    
    Features:
    - Fast similarity search using FAISS IndexFlatIP (inner product)
    - In-memory storage with disk persistence
    - Metadata support for document tracking
    - Automatic normalization for cosine similarity
    - Memory-efficient batch processing for large datasets
    """
    
    def __init__(self, embedding_dim: Optional[int] = None, batch_size: int = 1000):
        """Initialize the FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings to store. If None, will be
                          inferred from the first batch of embeddings added.
            batch_size: Maximum batch size for processing large datasets
        """
        # Store embedding dimension (will be set when first documents are added)
        self._embedding_dim = embedding_dim
        
        # FAISS index for similarity search (initialized when first documents added)
        self._index: Optional[faiss.Index] = None
        
        # Store document texts and metadata parallel to FAISS index
        self._documents: List[str] = []  # Document texts indexed by FAISS position
        self._metadata: List[Dict[str, Any]] = []  # Document metadata indexed by FAISS position
        self._document_ids: List[str] = []  # Unique IDs for each document
        
        # Counter for generating unique document IDs
        self._next_doc_id = 0
        
        # Memory management settings
        self._batch_size = batch_size  # Batch size for large dataset processing
        self._total_documents = 0  # Track total documents for memory estimation


    def add_documents(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Add documents with their embeddings to the FAISS vector store.
        
        Args:
            texts: List of document text chunks to store
            embeddings: Numpy array of shape (n_documents, embedding_dim) with document embeddings
            metadata: Optional list of metadata dictionaries for each document
            batch_size: Override default batch size for this operation
            
        Returns:
            List of unique document IDs assigned to the added documents
        """
        # Use instance batch size if not provided
        effective_batch_size = batch_size or self._batch_size
        
        # Process large datasets in batches to manage memory
        if len(texts) > effective_batch_size:
            return self._add_documents_in_batches(texts, embeddings, metadata, effective_batch_size)
        
        # Validate input parameters
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy array")
        
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"Number of texts ({len(texts)}) must match number of embeddings ({embeddings.shape[0]})")
        
        # Set embedding dimension if not already set
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        # Validate embedding dimensions match expected
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self._embedding_dim}")
        
        # Initialize FAISS index if this is the first batch of documents
        if self._index is None:
            # Use IndexFlatIP for inner product similarity (cosine when normalized)
            self._index = faiss.IndexFlatIP(self._embedding_dim)

        # Normalize embeddings for cosine similarity search
        embeddings_normalized = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)  # In-place normalization
        
        # Add embeddings to FAISS index
        self._index.add(embeddings_normalized)
        
        # Generate unique document IDs for the new documents
        new_doc_ids = []
        for i in range(len(texts)):
            doc_id = f"doc_{self._next_doc_id}"
            new_doc_ids.append(doc_id)
            self._next_doc_id += 1
        
        # Store document texts, metadata, and IDs
        self._documents.extend(texts)
        self._document_ids.extend(new_doc_ids)
        
        # Handle metadata (use empty dict if not provided)
        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of texts ({len(texts)})")
        
        self._metadata.extend(metadata)
        
        # Update total documents count
        self._total_documents += len(texts)

        return new_doc_ids

    def _add_documents_in_batches(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]],
        batch_size: int
    ) -> List[str]:
        """Add documents in batches to manage memory efficiently."""
        all_doc_ids = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            
            # Extract batch data
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadata = metadata[i:end_idx] if metadata else None
            
            # Process batch
            batch_doc_ids = self.add_documents(
                batch_texts, 
                batch_embeddings, 
                batch_metadata,
                batch_size=batch_size  # Prevent recursive batching
            )
            all_doc_ids.extend(batch_doc_ids)
        
        return all_doc_ids

    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Get estimated memory usage of the vector store."""
        if self._embedding_dim is None:
            return {'status': 'no_data', 'estimate_mb': 0}
        
        # Estimate memory usage
        embedding_memory = self._total_documents * self._embedding_dim * 4  # 4 bytes per float32
        text_memory = sum(len(text.encode('utf-8')) for text in self._documents)
        metadata_memory = len(str(self._metadata).encode('utf-8'))
        
        total_mb = (embedding_memory + text_memory + metadata_memory) / (1024 * 1024)
        
        return {
            'status': 'estimated',
            'total_documents': self._total_documents,
            'embedding_dimension': self._embedding_dim,
            'estimate_mb': round(total_mb, 2),
            'embedding_memory_mb': round(embedding_memory / (1024 * 1024), 2),
            'text_memory_mb': round(text_memory / (1024 * 1024), 2),
            'metadata_memory_mb': round(metadata_memory / (1024 * 1024), 2)
        }

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
        if self._index is None or len(self._documents) == 0:
            raise RuntimeError("No documents have been added to the vector store")
        
        # Validate query embedding dimensions
        if query_embedding.shape != (self._embedding_dim,):
            raise ValueError(f"Query embedding shape {query_embedding.shape} doesn't match expected ({self._embedding_dim},)")
        
        # Normalize query embedding for cosine similarity
        query_normalized = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Perform similarity search using FAISS
        # Note: FAISS returns squared L2 distances, but we normalized so this gives us cosine similarity
        search_k = min(top_k, len(self._documents))  # Don't search for more docs than we have
        similarities, indices = self._index.search(query_normalized, search_k)
        
        # Extract results from FAISS search
        similarities = similarities[0]  # Remove batch dimension
        indices = indices[0]  # Remove batch dimension
        
        # Collect results from document storage
        result_texts = []
        result_scores = []
        result_metadata = []
        
        for i, idx in enumerate(indices):
            # Skip invalid indices (FAISS can return -1 for no match)
            if idx == -1:
                continue
                
            # Get document data
            doc_text = self._documents[idx]
            doc_metadata = self._metadata[idx].copy()
            similarity_score = float(similarities[i])
            
            # Apply metadata filter if provided
            if metadata_filter is not None:
                # Check if all filter criteria match the document metadata
                matches_filter = all(
                    key in doc_metadata and doc_metadata[key] == value
                    for key, value in metadata_filter.items()
                )
                if not matches_filter:
                    continue  # Skip this document
            
            # Add to results
            result_texts.append(doc_text)
            result_scores.append(similarity_score)
            result_metadata.append(doc_metadata)
            
            # Stop if we have enough results
            if len(result_texts) >= top_k:
                break
        

        return result_texts, result_scores, result_metadata

    def get_document_count(self) -> int:
        """Get the total number of documents stored in the vector database."""
        return len(self._documents)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings stored in this vector store."""
        return self._embedding_dim

    def save(self, filepath: str) -> None:
        """Save the FAISS vector store to disk for persistence.
        
        Args:
            filepath: Path where the vector store should be saved (without extension)
        """
        if self._index is None:
            raise RuntimeError("Cannot save empty vector store")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, f"{filepath}.faiss")
        
        # Save metadata and documents using pickle
        store_data = {
            'embedding_dim': self._embedding_dim,
            'documents': self._documents,
            'metadata': self._metadata,
            'document_ids': self._document_ids,
            'next_doc_id': self._next_doc_id
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(store_data, f)
        


    def load(self, filepath: str) -> None:
        """Load a previously saved FAISS vector store from disk.
        
        Args:
            filepath: Path to the saved vector store file (without extension)
        """
        # Check if files exist
        faiss_file = f"{filepath}.faiss"
        metadata_file = f"{filepath}.pkl"
        
        if not os.path.exists(faiss_file):
            raise FileNotFoundError(f"FAISS index file not found: {faiss_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load FAISS index
        self._index = faiss.read_index(faiss_file)
        
        # Load metadata and documents
        with open(metadata_file, 'rb') as f:
            store_data = pickle.load(f)
        
        self._embedding_dim = store_data['embedding_dim']
        self._documents = store_data['documents']
        self._metadata = store_data['metadata']
        self._document_ids = store_data['document_ids']
        self._next_doc_id = store_data['next_doc_id']
        


    def clear(self) -> None:
        """Remove all documents from the vector store."""
        self._index = None
        self._documents.clear()
        self._metadata.clear()
        self._document_ids.clear()
        self._next_doc_id = 0
 