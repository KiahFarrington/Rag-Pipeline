"""Hybrid retriever combining dense vectors with sparse keyword matching.

This retriever uses both vector similarity and keyword overlap to find
relevant chunks. It's more robust for queries that need both semantic
and exact keyword matching.
"""

from typing import List, Tuple, Set
import numpy as np
import re
from collections import Counter
from app.vector_db.base_vector_store import BaseVectorStore


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval.
    
    Uses both vector similarity and keyword matching to provide
    more comprehensive search results for RAG systems.
    """
    
    def __init__(
        self, 
        vector_weight: float = 0.7, 
        keyword_weight: float = 0.3,
        min_keyword_overlap: int = 1
    ):
        """Initialize hybrid retriever.
        
        Parameters
        ----------
        vector_weight : float, optional
            Weight for vector similarity scores (default: 0.7)
        keyword_weight : float, optional
            Weight for keyword overlap scores (default: 0.3)
        min_keyword_overlap : int, optional
            Minimum keyword overlap required (default: 1)
        """
        # Validate weights sum to 1.0
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError("vector_weight + keyword_weight must equal 1.0")
        
        # Store retrieval parameters
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.min_keyword_overlap = min_keyword_overlap
        
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text for matching.
        
        Parameters
        ----------
        text : str
            Input text to extract keywords from
            
        Returns
        -------
        Set[str]
            Set of lowercase keywords
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 
            'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Return keywords without stop words
        return {word for word in words if word not in stop_words and len(word) > 2}
    
    def _calculate_keyword_score(self, query_keywords: Set[str], chunk: str) -> float:
        """Calculate keyword overlap score between query and chunk.
        
        Parameters
        ----------
        query_keywords : Set[str]
            Keywords extracted from query
        chunk : str
            Chunk text to score
            
        Returns
        -------
        float
            Keyword overlap score (0.0 to 1.0)
        """
        # Extract keywords from chunk
        chunk_keywords = self._extract_keywords(chunk)
        
        # Calculate overlap
        if not query_keywords:
            return 0.0
            
        # Find intersection of keywords
        overlapping_keywords = query_keywords.intersection(chunk_keywords)
        
        # Calculate score as ratio of overlapping keywords
        overlap_ratio = len(overlapping_keywords) / len(query_keywords)
        
        return overlap_ratio
    
    def retrieve(
        self, 
        vector_store: BaseVectorStore, 
        query_embedding: np.ndarray, 
        query_text: str,
        top_k: int = 5,
        candidate_multiplier: int = 3
    ) -> List[str]:
        """Retrieve chunks using hybrid dense + sparse approach.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing chunk embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        query_text : str
            Original query text for keyword extraction
        top_k : int, optional
            Number of final chunks to retrieve (default: 5)
        candidate_multiplier : int, optional
            Multiplier for initial candidates (default: 3)
            
        Returns
        -------
        List[str]
            List of relevant chunks ordered by hybrid score
        """
        # Validate inputs early
        if query_embedding is None or len(query_embedding) == 0:
            return []
            
        if not query_text or not query_text.strip():
            return []
            
        if top_k <= 0:
            return []
        
        # Step 1: Get more candidates from vector similarity
        candidate_k = min(top_k * candidate_multiplier, 50)  # Cap at 50 candidates
        chunks, vector_scores, metadata = vector_store.search(query_embedding, candidate_k)
        
        # Handle case with no results
        if not chunks:
            return []
        
        # Step 2: Extract query keywords for sparse matching
        query_keywords = self._extract_keywords(query_text)
        
        # Step 3: Calculate hybrid scores for each candidate
        hybrid_scores = []
        for i, chunk in enumerate(chunks):
            # Get normalized vector score (assume scores are already 0-1)
            vector_score = vector_scores[i] if i < len(vector_scores) else 0.0
            
            # Calculate keyword overlap score
            keyword_score = self._calculate_keyword_score(query_keywords, chunk)
            
            # Combine scores with weights
            hybrid_score = (self.vector_weight * vector_score + 
                          self.keyword_weight * keyword_score)
            
            # Only include if meets minimum keyword overlap
            keyword_overlap_count = len(self._extract_keywords(chunk).intersection(query_keywords))
            if keyword_overlap_count >= self.min_keyword_overlap:
                hybrid_scores.append((chunk, hybrid_score))
        
        # Step 4: Sort by hybrid score and return top_k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the chunks (not scores)
        return [chunk for chunk, score in hybrid_scores[:top_k]]
    
    def retrieve_with_scores(
        self, 
        vector_store: BaseVectorStore, 
        query_embedding: np.ndarray, 
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """Retrieve chunks with detailed scoring information.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing chunk embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        query_text : str
            Original query text for keyword extraction
        top_k : int, optional
            Number of chunks to retrieve (default: 5)
            
        Returns
        -------
        List[Tuple[str, float, dict]]
            List of (chunk, hybrid_score, score_breakdown) tuples
        """
        # Use similar logic as retrieve() but return detailed scores
        # This would include vector_score, keyword_score, hybrid_score in the dict
        # Implementation similar to retrieve() above but with more detail
        pass  # Placeholder for now


# Convenience function for easy usage
def retrieve_hybrid_chunks(
    vector_store: BaseVectorStore, 
    query_embedding: np.ndarray, 
    query_text: str,
    top_k: int = 5
) -> List[str]:
    """Convenience function for hybrid retrieval.
    
    Uses default hybrid retriever settings to find relevant chunks.
    """
    # Create retriever with default settings and use it
    retriever = HybridRetriever()
    return retriever.retrieve(vector_store, query_embedding, query_text, top_k) 