"""Advanced retriever with re-ranking and query expansion.

This retriever provides enhanced search capabilities including:
- Query expansion for better recall
- Result re-ranking based on multiple factors
- Contextual filtering for improved precision
- Diversity-aware selection to avoid redundant results
"""

import re  # Regular expressions for text processing
import logging  # Logging system for debugging
from typing import List, Tuple, Dict, Any, Set, Optional  # Type hints for clarity
from collections import Counter  # For frequency counting
import numpy as np  # Numerical operations
from .dense_retriever import DenseRetriever  # Base dense retrieval
from .hybrid_retriever import HybridRetriever  # Hybrid retrieval
from app.vector_db.base_vector_store import BaseVectorStore  # Vector store interface

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance


class AdvancedRetriever:
    """Advanced retriever with re-ranking and query enhancement capabilities."""
    
    def __init__(
        self,
        base_retriever_type: str = 'hybrid',  # Base retriever to use
        rerank_results: bool = True,  # Whether to re-rank results
        expand_queries: bool = True,  # Whether to expand queries
        diversity_factor: float = 0.3,  # Factor for result diversification
        context_window: int = 2,  # Context window for relevance scoring
        min_relevance_score: float = 0.1  # Minimum relevance threshold
    ):
        """Initialize advanced retriever with configuration options.
        
        Parameters
        ----------
        base_retriever_type : str
            Type of base retriever ('dense', 'hybrid')
        rerank_results : bool
            Whether to apply result re-ranking
        expand_queries : bool
            Whether to apply query expansion
        diversity_factor : float
            Factor for promoting diversity in results (0-1)
        context_window : int
            Number of chunks to consider for context scoring
        min_relevance_score : float
            Minimum relevance score to include results
        """
        # Store configuration parameters
        self.base_retriever_type = base_retriever_type  # Base retriever type selection
        self.rerank_results = rerank_results  # Enable/disable re-ranking
        self.expand_queries = expand_queries  # Enable/disable query expansion
        self.diversity_factor = diversity_factor  # Diversity promotion factor
        self.context_window = context_window  # Context window size
        self.min_relevance_score = min_relevance_score  # Minimum relevance threshold
        
        # Initialize base retriever based on type
        if base_retriever_type == 'hybrid':
            self.base_retriever = HybridRetriever()  # Use hybrid retriever
        else:
            self.base_retriever = DenseRetriever()  # Use dense retriever
        
        # Log initialization
        logger.info(f"AdvancedRetriever initialized with {base_retriever_type} base, rerank={rerank_results}, expand={expand_queries}")
    
    def retrieve(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        candidate_multiplier: int = 4
    ) -> List[str]:
        """Retrieve relevant chunks using advanced retrieval techniques.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing document embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        query_text : str
            Original query text for processing
        top_k : int
            Number of final results to return
        candidate_multiplier : int
            Multiplier for initial candidate retrieval
            
        Returns
        -------
        List[str]
            List of relevant text chunks
        """
        try:
            # Step 1: Expand query if enabled
            expanded_queries = self._expand_query(query_text) if self.expand_queries else [query_text]
            
            # Step 2: Retrieve candidates using base retriever
            all_candidates = []  # Store all candidate results
            candidate_k = min(top_k * candidate_multiplier, 50)  # Limit total candidates
            
            for expanded_query in expanded_queries:
                # Get candidates for each expanded query
                if self.base_retriever_type == 'hybrid':
                    candidates = self.base_retriever.retrieve_with_scores(
                        vector_store, query_embedding, expanded_query, candidate_k
                    )
                else:
                    candidates = self.base_retriever.retrieve_with_scores(
                        vector_store, query_embedding, candidate_k
                    )
                
                # Check if candidates is not None before iterating
                if candidates:
                    # Add query info to candidates
                    for chunk, score in candidates:
                        all_candidates.append({
                            'chunk': chunk,  # Document chunk text
                            'base_score': score,  # Original retrieval score
                            'query': expanded_query,  # Query that retrieved this chunk
                            'embedding': query_embedding  # Query embedding for re-ranking
                        })
            
            # Step 3: Remove duplicates while preserving best scores
            unique_candidates = self._deduplicate_candidates(all_candidates)
            
            # Step 4: Re-rank results if enabled
            if self.rerank_results:
                scored_candidates = self._rerank_candidates(unique_candidates, query_text, vector_store)
            else:
                scored_candidates = [(cand['chunk'], cand['base_score']) for cand in unique_candidates]
            
            # Step 5: Apply diversity filtering
            final_results = self._apply_diversity_filtering(scored_candidates, top_k)
            
            # Step 6: Filter by minimum relevance score
            filtered_results = [
                chunk for chunk, score in final_results 
                if score >= self.min_relevance_score
            ]
            
            # Log retrieval statistics
            logger.info(f"Advanced retrieval: {len(expanded_queries)} queries → {len(unique_candidates)} candidates → {len(filtered_results)} final results")
            
            return filtered_results[:top_k]  # Return top results
            
        except Exception as e:
            # Fallback to base retriever on error
            logger.error(f"Advanced retrieval failed, falling back to base retriever: {str(e)}")
            try:
                if self.base_retriever_type == 'hybrid':
                    fallback_results = self.base_retriever.retrieve_with_scores(vector_store, query_embedding, query_text, top_k)
                else:
                    fallback_results = self.base_retriever.retrieve_with_scores(vector_store, query_embedding, top_k)
                return fallback_results if fallback_results else []  # Ensure we return a list
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []  # Return empty list instead of None
    
    def retrieve_with_scores(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks with advanced scoring.
        
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store containing document embeddings
        query_embedding : np.ndarray
            Dense embedding of the query
        query_text : str
            Original query text for processing
        top_k : int
            Number of results to return
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (chunk, score) tuples with advanced scores
        """
        try:
            # Follow same process as retrieve() but return scores
            expanded_queries = self._expand_query(query_text) if self.expand_queries else [query_text]
            
            all_candidates = []  # Store all candidate results
            candidate_k = min(top_k * 4, 50)  # Limit candidates
            
            for expanded_query in expanded_queries:
                if self.base_retriever_type == 'hybrid':
                    candidates = self.base_retriever.retrieve_with_scores(
                        vector_store, query_embedding, expanded_query, candidate_k
                    )
                else:
                    candidates = self.base_retriever.retrieve_with_scores(
                        vector_store, query_embedding, candidate_k
                    )
                
                # Check if candidates is not None before iterating
                if candidates:
                    for chunk, score in candidates:
                        all_candidates.append({
                            'chunk': chunk,
                            'base_score': score,
                            'query': expanded_query,
                            'embedding': query_embedding
                        })
            
            # Process candidates
            unique_candidates = self._deduplicate_candidates(all_candidates)
            
            if self.rerank_results:
                scored_candidates = self._rerank_candidates(unique_candidates, query_text, vector_store)
            else:
                scored_candidates = [(cand['chunk'], cand['base_score']) for cand in unique_candidates]
            
            # Apply diversity and relevance filtering
            final_results = self._apply_diversity_filtering(scored_candidates, top_k)
            filtered_results = [
                (chunk, score) for chunk, score in final_results 
                if score >= self.min_relevance_score
            ]
            
            return filtered_results[:top_k]  # Return results with scores
            
        except Exception as e:
            # Fallback to base retriever
            logger.error(f"Advanced retrieval with scores failed: {str(e)}")
            try:
                if self.base_retriever_type == 'hybrid':
                    fallback_results = self.base_retriever.retrieve_with_scores(vector_store, query_embedding, query_text, top_k)
                else:
                    fallback_results = self.base_retriever.retrieve_with_scores(vector_store, query_embedding, top_k)
                return fallback_results if fallback_results else []  # Ensure we return a list
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval with scores also failed: {str(fallback_error)}")
                return []  # Return empty list instead of None
    
    def _expand_query(self, query_text: str) -> List[str]:
        """Expand query with synonyms and related terms, optimized for procedural content.
        
        Parameters
        ----------
        query_text : str
            Original query text
            
        Returns
        -------
        List[str]
            List of expanded queries including original
        """
        expanded_queries = [query_text]  # Start with original query
        
        # Detect if this is a procedural query
        is_procedural = self._is_procedural_query(query_text)
        
        # Technical domain expansions
        expansions = {
            'protocol': ['communication protocol', 'network protocol', 'data protocol'],
            'security': ['cybersecurity', 'data security', 'network security'],
            'communication': ['data transmission', 'messaging', 'networking'],
            'configuration': ['setup', 'installation', 'settings'],
            'authentication': ['login', 'access control', 'verification'],
            'encryption': ['cryptography', 'data protection', 'secure transmission'],
            'network': ['networking', 'connectivity', 'infrastructure'],
            'system': ['platform', 'architecture', 'framework'],
            'data': ['information', 'content', 'payload'],
            'control': ['management', 'administration', 'governance']
        }
        
        # Procedural/installation specific expansions
        procedural_expansions = {
            'install': ['installation', 'setup', 'mount', 'connect', 'configure'],
            'setup': ['installation', 'configure', 'prepare', 'initialize'],
            'steps': ['procedure', 'instructions', 'process', 'method'],
            'how to': ['procedure for', 'steps to', 'method to', 'process to'],
            'power supply': ['PSU', 'power unit', 'power source'],
            'controllogix': ['control logix', 'allen bradley', 'plc', 'controller'],
            'chassis': ['rack', 'enclosure', 'mounting frame'],
            'procedure': ['steps', 'instructions', 'process', 'method'],
            'connect': ['attach', 'fasten', 'secure', 'join'],
            'mount': ['install', 'attach', 'secure', 'fasten']
        }
        
        # Check for expandable terms in query
        query_lower = query_text.lower()  # Convert to lowercase for matching
        
        # Apply technical expansions
        for term, synonyms in expansions.items():
            if term in query_lower:
                # Add queries with synonyms
                for synonym in synonyms:
                    expanded_query = query_text.replace(term, synonym)  # Replace term with synonym
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)  # Add unique expansion
        
        # Apply procedural expansions if this is a procedural query
        if is_procedural:
            for term, synonyms in procedural_expansions.items():
                if term in query_lower:
                    # Add queries with synonyms
                    for synonym in synonyms:
                        expanded_query = query_text.replace(term, synonym)  # Replace term with synonym
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)  # Add unique expansion
            
            # Add procedural keywords to help find step-by-step content
            procedural_keywords = [
                f"step {query_text}",
                f"procedure {query_text}",
                f"instructions {query_text}",
                f"{query_text} steps",
                f"{query_text} procedure"
            ]
            for keyword_query in procedural_keywords:
                if keyword_query not in expanded_queries:
                    expanded_queries.append(keyword_query)
        
        # Limit number of expansions to avoid overwhelming retrieval
        max_expansions = 6 if is_procedural else 3  # More expansions for procedural queries
        if len(expanded_queries) > max_expansions:
            expanded_queries = expanded_queries[:max_expansions]  # Limit expansions
        
        logger.debug(f"Query expansion: '{query_text}' → {len(expanded_queries)} variants (procedural: {is_procedural})")
        
        return expanded_queries  # Return expanded query list
    
    def _is_procedural_query(self, query: str) -> bool:
        """Detect if query is asking for steps, procedures, or instructions.
        
        Parameters
        ----------
        query : str
            User's question
            
        Returns
        -------
        bool
            True if query appears to be asking for procedural information
        """
        # Keywords that indicate procedural queries
        procedural_keywords = [
            r'\bsteps?\b',  # "step", "steps"
            r'\bhow\s+to\b',  # "how to"
            r'\binstall\b',  # "install"
            r'\bsetup\b',  # "setup"
            r'\bconfigure\b',  # "configure"
            r'\bprocedure\b',  # "procedure"
            r'\binstructions?\b',  # "instruction", "instructions"
            r'\bprocess\b',  # "process"
            r'\bmethod\b',  # "method"
            r'\bway\s+to\b',  # "way to"
            r'\bguide\b',  # "guide"
            r'\btutorial\b',  # "tutorial"
        ]
        
        # Check if query contains any procedural keywords
        for pattern in procedural_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                return True  # Query appears to be procedural
        
        return False  # Query doesn't appear to be procedural
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates while keeping best scores.
        
        Parameters
        ----------
        candidates : List[Dict[str, Any]]
            List of candidate dictionaries
            
        Returns
        -------
        List[Dict[str, Any]]
            Deduplicated candidates with best scores
        """
        # Group candidates by chunk text
        chunk_groups = {}  # Dictionary to group by chunk text
        
        for candidate in candidates:
            chunk_text = candidate['chunk']  # Get chunk text
            
            if chunk_text not in chunk_groups:
                chunk_groups[chunk_text] = []  # Initialize group for this chunk
            
            chunk_groups[chunk_text].append(candidate)  # Add candidate to group
        
        # Keep best candidate from each group
        unique_candidates = []  # List for unique candidates
        
        for chunk_text, group in chunk_groups.items():
            # Find candidate with highest base score
            best_candidate = max(group, key=lambda x: x['base_score'])  # Get best scoring candidate
            unique_candidates.append(best_candidate)  # Add to unique list
        
        # Sort by base score
        unique_candidates.sort(key=lambda x: x['base_score'], reverse=True)  # Sort by score
        
        logger.debug(f"Deduplication: {len(candidates)} → {len(unique_candidates)} unique candidates")
        
        return unique_candidates  # Return deduplicated candidates
    
    def _rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        original_query: str,
        vector_store: BaseVectorStore
    ) -> List[Tuple[str, float]]:
        """Re-rank candidates using multiple scoring factors.
        
        Parameters
        ----------
        candidates : List[Dict[str, Any]]
            Candidate results to re-rank
        original_query : str
            Original user query
        vector_store : BaseVectorStore
            Vector store for context retrieval
            
        Returns
        -------
        List[Tuple[str, float]]
            Re-ranked results with new scores
        """
        reranked_results = []  # List for re-ranked results
        
        # Check if this is a procedural query
        is_procedural = self._is_procedural_query(original_query)
        
        for candidate in candidates:
            chunk = candidate['chunk']  # Get chunk text
            base_score = candidate['base_score']  # Get base retrieval score
            
            # Calculate additional scoring factors
            query_match_score = self._calculate_query_match_score(chunk, original_query)  # Query term matching
            length_score = self._calculate_length_score(chunk)  # Chunk length appropriateness
            context_score = self._calculate_context_score(chunk, candidates)  # Context relevance
            
            # Calculate procedural content score if applicable
            procedural_score = 0.0
            if is_procedural:
                procedural_score = self._calculate_procedural_score(chunk)  # Score for step-by-step content
            
            # Combine scores with weights (adjusted for procedural queries)
            if is_procedural:
                final_score = (
                    0.3 * base_score +  # Base retrieval score (30%)
                    0.25 * query_match_score +  # Direct query matching (25%)
                    0.25 * procedural_score +  # Procedural content bonus (25%)
                    0.15 * context_score +  # Context relevance (15%)
                    0.05 * length_score  # Length appropriateness (5%)
                )
            else:
                final_score = (
                    0.4 * base_score +  # Base retrieval score (40%)
                    0.3 * query_match_score +  # Direct query matching (30%)
                    0.2 * context_score +  # Context relevance (20%)
                    0.1 * length_score  # Length appropriateness (10%)
                )
            
            reranked_results.append((chunk, final_score))  # Add re-ranked result
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x[1], reverse=True)  # Sort by final score
        
        logger.debug(f"Re-ranking completed for {len(candidates)} candidates")
        
        return reranked_results  # Return re-ranked results
    
    def _calculate_query_match_score(self, chunk: str, query: str) -> float:
        """Calculate direct query term matching score."""
        # Extract important terms from query (remove stop words)
        query_terms = self._extract_important_terms(query)  # Get important query terms
        chunk_terms = self._extract_important_terms(chunk)  # Get important chunk terms
        
        if not query_terms:
            return 0.0  # No terms to match
        
        # Calculate term overlap
        matching_terms = query_terms.intersection(chunk_terms)  # Find common terms
        match_ratio = len(matching_terms) / len(query_terms)  # Calculate match ratio
        
        # Boost score for exact phrase matches
        phrase_bonus = 0.0  # Initialize phrase bonus
        if len(query.split()) > 1:  # Multi-word query
            if query.lower() in chunk.lower():
                phrase_bonus = 0.5  # Bonus for exact phrase match
        
        total_score = min(1.0, match_ratio + phrase_bonus)  # Combine scores, cap at 1.0
        return total_score  # Return query match score
    
    def _calculate_length_score(self, chunk: str) -> float:
        """Calculate score based on chunk length appropriateness."""
        # Optimal length range for good context
        optimal_min = 200  # Minimum optimal length
        optimal_max = 1500  # Maximum optimal length
        
        chunk_length = len(chunk)  # Get chunk length
        
        if optimal_min <= chunk_length <= optimal_max:
            return 1.0  # Perfect length score
        elif chunk_length < optimal_min:
            return chunk_length / optimal_min  # Penalize short chunks
        else:
            return optimal_max / chunk_length  # Penalize very long chunks
    
    def _calculate_context_score(self, chunk: str, all_candidates: List[Dict[str, Any]]) -> float:
        """Calculate contextual relevance score."""
        # Extract key terms from this chunk
        chunk_terms = self._extract_important_terms(chunk)  # Get chunk terms
        
        # Compare with other high-scoring candidates
        context_scores = []  # List for context scores
        
        for candidate in all_candidates[:5]:  # Check top 5 candidates
            if candidate['chunk'] == chunk:
                continue  # Skip self-comparison
            
            other_terms = self._extract_important_terms(candidate['chunk'])  # Get other chunk terms
            
            # Calculate term overlap
            if chunk_terms and other_terms:
                overlap = len(chunk_terms.intersection(other_terms))  # Count overlapping terms
                overlap_ratio = overlap / len(chunk_terms.union(other_terms))  # Calculate overlap ratio
                context_scores.append(overlap_ratio)  # Add to context scores
        
        # Return average context score
        return sum(context_scores) / len(context_scores) if context_scores else 0.0  # Average context score
    
    def _calculate_procedural_score(self, chunk: str) -> float:
        """Calculate score based on procedural/step-by-step content indicators.
        
        Parameters
        ----------
        chunk : str
            Text chunk to score
            
        Returns
        -------
        float
            Score from 0.0 to 1.0 indicating procedural content quality
        """
        score = 0.0  # Initialize score
        
        # Step indicators (strong indicators of procedural content)
        step_patterns = [
            r'\bStep\s+\d+[:.]\s*',  # "Step 1:", "Step 2."
            r'\b\d+\.\s+',  # "1. ", "2. "
            r'\b\d+\)\s*',  # "1) ", "2) "
            r'\bProcedure\s+\d+[:.]\s*',  # "Procedure 1:"
        ]
        
        # Count step indicators
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, chunk, re.IGNORECASE))
        
        # Base score from step indicators (0.4 max)
        if step_count > 0:
            score += min(0.4, step_count * 0.15)  # 0.15 per step, cap at 0.4
        
        # Action verbs (medium indicators)
        action_patterns = [
            r'\b(?:install|Install|INSTALL)\b',  # Installation terms
            r'\b(?:connect|Connect|CONNECT)\b',  # Connection terms
            r'\b(?:remove|Remove|REMOVE)\b',  # Removal terms
            r'\b(?:place|Place|PLACE)\b',  # Placement terms
            r'\b(?:secure|Secure|SECURE)\b',  # Securing terms
            r'\b(?:verify|Verify|VERIFY)\b',  # Verification terms
            r'\b(?:check|Check|CHECK)\b',  # Checking terms
            r'\b(?:ensure|Ensure|ENSURE)\b',  # Ensuring terms
        ]
        
        # Count action verbs
        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, chunk))
        
        # Action verb score (0.2 max)
        if action_count > 0:
            score += min(0.2, action_count * 0.05)  # 0.05 per action, cap at 0.2
        
        # Warning/safety indicators (important for technical procedures)
        safety_patterns = [
            r'\b(?:warning|Warning|WARNING)\b',  # Warning terms
            r'\b(?:caution|Caution|CAUTION)\b',  # Caution terms
            r'\b(?:important|Important|IMPORTANT)\b',  # Important notices
            r'\b(?:note|Note|NOTE)\b',  # Notes
            r'\b(?:danger|Danger|DANGER)\b',  # Danger warnings
        ]
        
        # Count safety indicators
        safety_count = 0
        for pattern in safety_patterns:
            safety_count += len(re.findall(pattern, chunk))
        
        # Safety indicator score (0.15 max)
        if safety_count > 0:
            score += min(0.15, safety_count * 0.05)  # 0.05 per safety note, cap at 0.15
        
        # Sequential indicators (weak but helpful)
        sequential_patterns = [
            r'\b(?:first|First|FIRST)\b',  # First step
            r'\b(?:next|Next|NEXT)\b',  # Next step
            r'\b(?:then|Then|THEN)\b',  # Then step
            r'\b(?:finally|Finally|FINALLY)\b',  # Final step
            r'\b(?:before|Before|BEFORE)\b',  # Before action
            r'\b(?:after|After|AFTER)\b',  # After action
        ]
        
        # Count sequential indicators
        sequential_count = 0
        for pattern in sequential_patterns:
            sequential_count += len(re.findall(pattern, chunk))
        
        # Sequential indicator score (0.1 max)
        if sequential_count > 0:
            score += min(0.1, sequential_count * 0.02)  # 0.02 per sequential word, cap at 0.1
        
        # Tools/equipment indicators (relevant for installation procedures)
        equipment_patterns = [
            r'\b(?:tools?|Tools?|TOOLS?)\b',  # Tools
            r'\b(?:screwdriver|wrench|pliers)\b',  # Specific tools
            r'\b(?:cable|wire|connector)\b',  # Connection items
            r'\b(?:bolt|screw|nut)\b',  # Fasteners
            r'\b(?:power\s+supply|PSU)\b',  # Power equipment
        ]
        
        # Count equipment indicators
        equipment_count = 0
        for pattern in equipment_patterns:
            equipment_count += len(re.findall(pattern, chunk, re.IGNORECASE))
        
        # Equipment indicator score (0.15 max)
        if equipment_count > 0:
            score += min(0.15, equipment_count * 0.03)  # 0.03 per equipment mention, cap at 0.15
        
        # Ensure score doesn't exceed 1.0
        final_score = min(1.0, score)
        
        logger.debug(f"Procedural score for chunk: {final_score:.3f} (steps: {step_count}, actions: {action_count}, safety: {safety_count})")
        
        return final_score  # Return procedural content score
    
    def _extract_important_terms(self, text: str) -> Set[str]:
        """Extract important terms from text (excluding stop words)."""
        # Basic stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }
        
        # Extract words using regex
        words = re.findall(r'\b\w+\b', text.lower())  # Extract words in lowercase
        
        # Filter important terms
        important_terms = {
            word for word in words 
            if len(word) > 2 and word not in stop_words  # Keep words longer than 2 chars, not stop words
        }
        
        return important_terms  # Return set of important terms
    
    def _apply_diversity_filtering(self, scored_candidates: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        """Apply diversity filtering to avoid redundant results."""
        if self.diversity_factor == 0 or len(scored_candidates) <= top_k:
            return scored_candidates[:top_k]  # No diversity filtering needed
        
        diverse_results = []  # List for diverse results
        remaining_candidates = scored_candidates.copy()  # Copy candidates list
        
        # Always include the top result
        if remaining_candidates:
            diverse_results.append(remaining_candidates.pop(0))  # Add top result
        
        # Add diverse results
        while len(diverse_results) < top_k and remaining_candidates:
            best_candidate = None  # Best candidate for diversity
            best_diversity_score = -1  # Best diversity score
            
            for i, (candidate_chunk, candidate_score) in enumerate(remaining_candidates):
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(
                    candidate_chunk, [chunk for chunk, _ in diverse_results]
                )
                
                # Combine relevance and diversity
                combined_score = (
                    (1 - self.diversity_factor) * candidate_score +  # Relevance component
                    self.diversity_factor * diversity_score  # Diversity component
                )
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score  # Update best score
                    best_candidate = (candidate_chunk, candidate_score)  # Update best candidate
                    best_index = i  # Track index for removal
            
            if best_candidate:
                diverse_results.append(best_candidate)  # Add diverse candidate
                remaining_candidates.pop(best_index)  # Remove from remaining
        
        logger.debug(f"Diversity filtering: applied factor {self.diversity_factor} to select {len(diverse_results)} diverse results")
        
        return diverse_results  # Return diverse results
    
    def _calculate_diversity_score(self, candidate_chunk: str, selected_chunks: List[str]) -> float:
        """Calculate diversity score for a candidate relative to selected chunks."""
        if not selected_chunks:
            return 1.0  # Maximum diversity if no selections yet
        
        candidate_terms = self._extract_important_terms(candidate_chunk)  # Get candidate terms
        
        # Calculate minimum similarity to any selected chunk
        min_similarity = float('inf')  # Initialize minimum similarity
        
        for selected_chunk in selected_chunks:
            selected_terms = self._extract_important_terms(selected_chunk)  # Get selected chunk terms
            
            # Calculate Jaccard similarity
            if candidate_terms or selected_terms:
                union_size = len(candidate_terms.union(selected_terms))  # Size of union
                intersection_size = len(candidate_terms.intersection(selected_terms))  # Size of intersection
                similarity = intersection_size / union_size if union_size > 0 else 0  # Jaccard similarity
                min_similarity = min(min_similarity, similarity)  # Track minimum similarity
        
        # Diversity is inverse of similarity
        diversity_score = 1.0 - min_similarity if min_similarity != float('inf') else 1.0
        
        return diversity_score  # Return diversity score


def get_retriever_info() -> Dict[str, Any]:
    """Get information about the advanced retriever.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing retriever information and capabilities
    """
    return {
        'method': 'advanced',  # Retriever method identifier
        'base_retrievers': ['dense', 'hybrid'],  # Available base retrievers
        'features': [
            'query_expansion',  # Query expansion capability
            'result_reranking',  # Result re-ranking capability
            'diversity_filtering',  # Diversity filtering capability
            'contextual_scoring'  # Contextual scoring capability
        ],
        'supports_scoring': True,  # Whether retriever supports scoring
        'configurable': True,  # Whether retriever is configurable
        'description': 'Advanced retriever with query expansion, re-ranking, and diversity filtering'
    } 