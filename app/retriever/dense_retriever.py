"""Dense retriever using pure vector similarity search.

This retriever finds relevant chunks based solely on dense vector embeddings
and cosine similarity. It's fast and works well for semantic queries.
"""

from typing import List, Tuple
import numpy as np
from vector_db.base_vector_store import BaseVectorStore


class DenseRetriever:
    """Dense vector similarity retriever.
    
    Uses dense embeddings and vector similarity to find relevant chunks.
    This is the standard approach for semantic search in RAG systems.
    """
    
    def __init__(self, similarity_threshold: float = 0.0):
        """Initialize dense retriever.
        
        Parameters
        ----------
        similarity_threshold : float, optional
            Minimum similarity score for results (default: 0.0)
        """
        # Store similarity threshold for filtering results
        self.similarity_threshold = similarity_threshold
        
    def retrieve(
        self, 
        vector_store: BaseVectorStore, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[str]:
        """Retrieve chunks using dense vector similarity.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing chunk embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        top_k : int, optional
            Number of chunks to retrieve (default: 5)
            
        Returns
        -------
        List[str]
            List of relevant chunks ordered by similarity
        """
        # Validate inputs early
        if query_embedding is None or len(query_embedding) == 0:
            return []
            
        if top_k <= 0:
            return []
        
        # Get search results from vector store
        chunks, scores, metadata = vector_store.search(query_embedding, top_k)
        
        # Filter by similarity threshold if set
        if self.similarity_threshold > 0.0:
            filtered_chunks = []
            for chunk, score in zip(chunks, scores):
                if score >= self.similarity_threshold:
                    filtered_chunks.append(chunk)
            return filtered_chunks
        
        # Return all chunks if no threshold
        return chunks
    
    def retrieve_with_scores(
        self, 
        vector_store: BaseVectorStore, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve chunks with their similarity scores.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing chunk embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        top_k : int, optional
            Number of chunks to retrieve (default: 5)
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (chunk, score) tuples ordered by similarity
        """
        # Validate inputs early
        if query_embedding is None or len(query_embedding) == 0:
            return []
            
        if top_k <= 0:
            return []
        
        # Get search results from vector store
        chunks, scores, metadata = vector_store.search(query_embedding, top_k)
        
        # Combine chunks and scores into tuples
        results = list(zip(chunks, scores))
        
        # Filter by similarity threshold if set
        if self.similarity_threshold > 0.0:
            results = [(chunk, score) for chunk, score in results 
                      if score >= self.similarity_threshold]
        
        return results


# Convenience function for backward compatibility
def retrieve_similar_chunks(
    vector_store: BaseVectorStore, 
    query_embedding: np.ndarray, 
    top_k: int = 5
) -> List[str]:
    """Legacy function - use DenseRetriever class for new code.
    
    Retrieves similar chunks using dense vector similarity.
    """
    # Create retriever instance and use it
    retriever = DenseRetriever()
    return retriever.retrieve(vector_store, query_embedding, top_k) 