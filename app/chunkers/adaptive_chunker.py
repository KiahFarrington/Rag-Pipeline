"""Adaptive text chunker - Combines semantic and size-based strategies.

This chunker intelligently adapts its strategy based on content analysis,
providing optimal chunk sizes for different types of documents.
"""

import re  # Regular expressions for text pattern matching
import logging  # Logging system for error tracking
from typing import List, Dict, Any, Optional  # Type hints for better code clarity
from .semantic_chunker import chunk_by_semantics, detect_text_structure  # Import semantic chunking
from .fixed_length_chunker import chunk_by_fixed_length, get_optimal_chunk_size  # Import fixed-length chunking

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance for this chunker


class AdaptiveChunker:
    """Adaptive chunker that selects optimal strategy based on content analysis."""
    
    def __init__(
        self,
        min_chunk_size: int = 100,  # Minimum size for any chunk
        max_chunk_size: int = 2000,  # Maximum size for any chunk
        target_chunk_size: int = 800,  # Preferred chunk size
        semantic_threshold: float = 0.6  # Threshold for using semantic chunking
    ):
        """Initialize adaptive chunker with configuration parameters.
        
        Parameters
        ----------
        min_chunk_size : int
            Minimum size for any chunk in characters
        max_chunk_size : int
            Maximum size for any chunk in characters
        target_chunk_size : int
            Preferred target size for chunks
        semantic_threshold : float
            Threshold for paragraph density to use semantic chunking
        """
        # Store configuration parameters
        self.min_chunk_size = min_chunk_size  # Minimum chunk size constraint
        self.max_chunk_size = max_chunk_size  # Maximum chunk size constraint
        self.target_chunk_size = target_chunk_size  # Target chunk size for optimization
        self.semantic_threshold = semantic_threshold  # Threshold for semantic vs fixed-length choice
        
        # Log initialization for debugging
        logger.info(f"AdaptiveChunker initialized with min={min_chunk_size}, max={max_chunk_size}, target={target_chunk_size}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using adaptive strategy based on content analysis.
        
        Parameters
        ----------
        text : str
            Input text to be chunked
            
        Returns
        -------
        List[str]
            List of optimally sized text chunks
            
        Raises
        ------
        ValueError
            If text is not a string
        RuntimeError
            If chunking process fails
        """
        # Early validation of input parameters
        if not isinstance(text, str):
            logger.error(f"Text must be string, got {type(text)}")  # Log type error
            raise ValueError(f"Text must be string, got {type(text)}")  # Raise type error
        
        # Handle empty input gracefully
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided to adaptive chunker")  # Log empty input
            return []  # Return empty list for empty input
        
        # Clean the input text
        clean_text = text.strip()  # Remove leading and trailing whitespace
        
        try:
            # Analyze text structure to determine best chunking strategy
            structure_info = self._analyze_text_structure(clean_text)  # Analyze document structure
            
            # Choose chunking strategy based on analysis
            strategy = self._select_chunking_strategy(structure_info)  # Select optimal strategy
            
            # Apply selected chunking strategy
            if strategy == 'semantic':
                chunks = self._apply_semantic_chunking(clean_text)  # Use semantic chunking
            elif strategy == 'hybrid':
                chunks = self._apply_hybrid_chunking(clean_text, structure_info)  # Use hybrid approach
            elif strategy == 'procedural':
                chunks = self._apply_procedural_chunking(clean_text)  # Use procedural chunking for installation docs
            else:
                chunks = self._apply_fixed_length_chunking(clean_text)  # Use fixed-length chunking
            
            # Post-process chunks to ensure size constraints
            optimized_chunks = self._optimize_chunk_sizes(chunks)  # Optimize chunk sizes
            
            # Log successful chunking completion
            logger.info(f"Adaptive chunking completed: {len(optimized_chunks)} chunks using {strategy} strategy")
            
            return optimized_chunks  # Return optimized chunks
            
        except Exception as e:
            # Handle any unexpected errors during chunking
            logger.error(f"Error during adaptive chunking: {str(e)}")  # Log chunking error
            raise RuntimeError(f"Adaptive chunking failed: {str(e)}")  # Raise runtime error with details
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure to determine optimal chunking strategy.
        
        Parameters
        ----------
        text : str
            Text to analyze
            
        Returns
        -------
        Dict[str, Any]
            Analysis results including paragraph density, average sentence length, etc.
        """
        # Use existing structure detection from semantic chunker
        base_structure = detect_text_structure(text)  # Get basic structure information
        
        # Calculate additional metrics for strategy selection
        paragraph_density = self._calculate_paragraph_density(text)  # Calculate paragraph break frequency
        average_sentence_length = self._calculate_average_sentence_length(text)  # Calculate sentence length
        complexity_score = self._calculate_complexity_score(text)  # Calculate text complexity
        
        # Combine all analysis results
        structure_info = {
            **base_structure,  # Include basic structure information
            'paragraph_density': paragraph_density,  # Add paragraph density metric
            'average_sentence_length': average_sentence_length,  # Add sentence length metric
            'complexity_score': complexity_score,  # Add complexity metric
            'document_type': self._detect_document_type(text)  # Detect document type (technical, narrative, etc.)
        }
        
        # Log analysis results for debugging
        logger.debug(f"Text structure analysis: {structure_info}")  # Log analysis results
        
        return structure_info  # Return comprehensive structure analysis
    
    def _calculate_paragraph_density(self, text: str) -> float:
        """Calculate the density of paragraph breaks in the text."""
        # Count paragraph breaks (double newlines or indented lines)
        paragraph_breaks = len(re.findall(r'\n\s*\n|\n\s{3,}', text))  # Count paragraph break patterns
        
        # Calculate density as breaks per 1000 characters
        if len(text) == 0:
            return 0.0  # Return zero for empty text
        
        density = (paragraph_breaks / len(text)) * 1000  # Calculate density per 1000 characters
        return density  # Return paragraph density metric
    
    def _calculate_average_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in characters."""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)  # Split on sentence-ending punctuation
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
        
        # Calculate average length
        if not sentences:
            return 0.0  # Return zero for no sentences
        
        total_length = sum(len(sentence) for sentence in sentences)  # Sum all sentence lengths
        average_length = total_length / len(sentences)  # Calculate average
        
        return average_length  # Return average sentence length
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity based on various factors."""
        # Count technical indicators
        technical_patterns = [
            r'\b\d+\.\d+\b',  # Version numbers (1.0, 2.3.4)
            r'\b[A-Z]{2,}\b',  # Acronyms (TCP, HTTP, API)
            r'\b\w+://\w+',  # URLs or protocols
            r'\b\w+\(\)',  # Function calls
            r'\[\w+\]',  # References or citations
        ]
        
        complexity_score = 0.0  # Initialize complexity score
        
        # Count matches for each pattern
        for pattern in technical_patterns:
            matches = len(re.findall(pattern, text))  # Count pattern matches
            complexity_score += matches  # Add to complexity score
        
        # Normalize by text length
        if len(text) > 0:
            complexity_score = (complexity_score / len(text)) * 1000  # Normalize per 1000 characters
        
        return complexity_score  # Return complexity score
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content patterns."""
        # Technical document indicators
        technical_indicators = [
            r'\bprotocol\b', r'\bAPI\b', r'\bHTTP\b', r'\bTCP\b',  # Protocol terms
            r'\binstall\b', r'\bconfigure\b', r'\bsetup\b',  # Installation terms
            r'\bstep\s*\d+', r'\d+\.\s', r'\b\d+\)\s',  # Numbered steps
            r'\bprocedure\b', r'\binstructions?\b', r'\bmanual\b'  # Procedural terms
        ]
        
        # Installation/procedure specific indicators
        installation_indicators = [
            r'\binstallation\b', r'\binstall\b', r'\bmount\b', r'\bconnect\b',
            r'\bwiring\b', r'\bcable\b', r'\bpower\s+supply\b', r'\bcontrollogix\b',
            r'\bchassis\b', r'\bstep\s*\d+', r'\bprocedure\b', r'\binstructions\b',
            r'\brequired\s+tools\b', r'\bsafety\b', r'\bwarning\b', r'\bcaution\b'
        ]
        
        # Academic/research indicators
        academic_indicators = [
            r'\babstract\b', r'\bintroduction\b', r'\bmethodology\b',
            r'\bresults\b', r'\bconclusion\b', r'\breferences?\b',
            r'\bcitation\b', r'\bfigure\s+\d+', r'\btable\s+\d+'
        ]
        
        # Legal/business indicators
        business_indicators = [
            r'\bcontract\b', r'\bagreement\b', r'\bpolicy\b',
            r'\bterms\b', r'\bconditions\b', r'\bliability\b',
            r'\bwarranty\b', r'\blegal\b'
        ]
        
        # Count indicators for each type
        technical_score = 0
        installation_score = 0
        academic_score = 0
        business_score = 0
        
        # Count technical indicators
        for pattern in technical_indicators:
            technical_score += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count installation indicators (subset of technical)
        for pattern in installation_indicators:
            installation_score += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count academic indicators
        for pattern in academic_indicators:
            academic_score += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Count business indicators
        for pattern in business_indicators:
            business_score += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Determine document type based on highest score
        max_score = max(technical_score, installation_score, academic_score, business_score)
        
        if max_score == 0:
            return 'general'  # No specific indicators found
        elif installation_score == max_score and installation_score >= 3:
            return 'installation'  # Installation/procedure document
        elif technical_score == max_score:
            return 'technical'  # General technical document
        elif academic_score == max_score:
            return 'academic'  # Academic/research document
        elif business_score == max_score:
            return 'business'  # Business/legal document
        else:
            return 'general'  # Default fallback
    
    def _select_chunking_strategy(self, structure_info: Dict[str, Any]) -> str:
        """Select optimal chunking strategy based on structure analysis."""
        # Extract key metrics
        paragraph_density = structure_info.get('paragraph_density', 0)  # Get paragraph density
        complexity_score = structure_info.get('complexity_score', 0)  # Get complexity score
        document_type = structure_info.get('document_type', 'general')  # Get document type
        total_length = structure_info.get('total_length', 0)  # Get total document length
        
        # Decision logic for strategy selection
        if document_type == 'installation':
            strategy = 'procedural'  # Use procedural chunking for installation/procedure documents
        elif paragraph_density >= self.semantic_threshold and document_type in ['academic', 'narrative']:
            strategy = 'semantic'  # Use semantic chunking for well-structured documents
        elif complexity_score > 5.0 and document_type == 'technical':
            strategy = 'hybrid'  # Use hybrid approach for complex technical documents
        elif total_length > 10000:
            strategy = 'hybrid'  # Use hybrid approach for very long documents
        else:
            strategy = 'fixed_length'  # Use fixed-length for simpler documents
        
        # Log strategy selection for debugging
        logger.info(f"Selected chunking strategy: {strategy} (density={paragraph_density:.2f}, complexity={complexity_score:.2f}, type={document_type})")
        
        return strategy  # Return selected strategy
    
    def _apply_semantic_chunking(self, text: str) -> List[str]:
        """Apply semantic chunking with size constraints."""
        # Use semantic chunker with minimum size constraint
        chunks = chunk_by_semantics(text, min_chunk_size=self.min_chunk_size)  # Get semantic chunks
        
        return chunks  # Return semantic chunks
    
    def _apply_fixed_length_chunking(self, text: str) -> List[str]:
        """Apply fixed-length chunking with optimal size."""
        # Use target chunk size for fixed-length chunking
        chunks = chunk_by_fixed_length(text, chunk_size=self.target_chunk_size)  # Get fixed-length chunks
        
        return chunks  # Return fixed-length chunks
    
    def _apply_hybrid_chunking(self, text: str, structure_info: Dict[str, Any]) -> List[str]:
        """Apply hybrid chunking combining semantic and size-based approaches."""
        # First try semantic chunking
        semantic_chunks = chunk_by_semantics(text, min_chunk_size=self.min_chunk_size // 2)  # Get semantic chunks with lower threshold
        
        # Post-process semantic chunks to ensure size constraints
        optimized_chunks = []  # List to store optimized chunks
        
        for chunk in semantic_chunks:
            # If chunk is too large, split it further
            if len(chunk) > self.max_chunk_size:
                # Split large chunk using fixed-length approach
                sub_chunks = chunk_by_fixed_length(chunk, chunk_size=self.target_chunk_size)  # Split large chunks
                optimized_chunks.extend(sub_chunks)  # Add sub-chunks to results
            elif len(chunk) >= self.min_chunk_size:
                optimized_chunks.append(chunk)  # Add chunk if it meets minimum size
        
        return optimized_chunks  # Return hybrid chunks
    
    def _apply_procedural_chunking(self, text: str) -> List[str]:
        """Apply procedural chunking to preserve step-by-step instructions and procedures.
        
        This method is optimized for installation manuals, technical procedures,
        and other documents where maintaining sequential steps is critical.
        """
        # Step 1: Identify procedural sections and step patterns
        step_patterns = [
            r'\bStep\s+\d+[:.]\s*',  # "Step 1:", "Step 2."
            r'\b\d+\.\s+',  # "1. ", "2. "
            r'\b\d+\)\s*',  # "1) ", "2) "
            r'\bProcedure\s+\d+[:.]\s*',  # "Procedure 1:"
            r'\bTask\s+\d+[:.]\s*',  # "Task 1:"
        ]
        
        # Step 2: Split text into logical sections
        section_separators = [
            r'\n\s*(?=Step\s+\d+)',  # Before step numbers
            r'\n\s*(?=\d+\.)',  # Before numbered lists
            r'\n\s*(?=STEP\s+\d+)',  # Before capitalized steps
            r'\n\s*(?=Procedure)',  # Before procedures
            r'\n\s*(?=PROCEDURE)',  # Before capitalized procedures
            r'\n\s*(?=WARNING[:\s])',  # Before warnings
            r'\n\s*(?=CAUTION[:\s])',  # Before cautions
            r'\n\s*(?=NOTE[:\s])',  # Before notes
            r'\n\s*(?=IMPORTANT[:\s])',  # Before important notices
        ]
        
        # Combine all separators into one pattern
        separator_pattern = '|'.join(f'({pattern})' for pattern in section_separators)
        
        # Split text into sections
        if re.search(separator_pattern, text, re.IGNORECASE | re.MULTILINE):
            sections = re.split(separator_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            # Remove empty sections and separator matches
            sections = [section.strip() for section in sections if section and section.strip() and not re.match(r'^\s*$', section)]
        else:
            # Fallback: split on double newlines for paragraph-based chunking
            sections = [section.strip() for section in text.split('\n\n') if section.strip()]
        
        # Step 3: Group related sections into coherent chunks
        chunks = []  # List to store final chunks
        current_chunk = ""  # Current chunk being built
        
        for section in sections:
            # Calculate potential chunk size if we add this section
            potential_size = len(current_chunk) + len(section) + 1  # +1 for separator
            
            # If adding this section would exceed max size, finalize current chunk
            if current_chunk and potential_size > self.max_chunk_size:
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())  # Add current chunk
                current_chunk = section  # Start new chunk with current section
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section  # Add with separator
                else:
                    current_chunk = section  # First section in chunk
            
            # Special handling for complete procedures or warnings
            if self._is_complete_procedure(section):
                # If this is a complete procedure, finalize the chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())  # Add complete procedure chunk
                    current_chunk = ""  # Start fresh
        
        # Add final chunk if it exists
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())  # Add final chunk
        
        # Step 4: Post-process chunks to ensure they contain meaningful content
        meaningful_chunks = []  # List for chunks with meaningful content
        
        for chunk in chunks:
            # Ensure chunk contains actual procedural content
            if self._contains_procedural_content(chunk):
                meaningful_chunks.append(chunk)  # Add chunk with procedural content
            elif len(chunk) >= self.min_chunk_size * 2:  # For large non-procedural chunks
                # Split using fixed-length approach
                sub_chunks = chunk_by_fixed_length(chunk, chunk_size=self.target_chunk_size)
                meaningful_chunks.extend(sub_chunks)  # Add sub-chunks
        
        # Log procedural chunking results
        logger.info(f"Procedural chunking created {len(meaningful_chunks)} chunks from {len(sections)} sections")
        
        return meaningful_chunks  # Return procedural chunks
    
    def _is_complete_procedure(self, text: str) -> bool:
        """Check if text contains a complete procedure or step sequence."""
        # Look for indicators of complete procedures
        completion_indicators = [
            r'\bcomplete\b.*\bprocedure\b',  # "complete procedure"
            r'\bfinish\b.*\binstallation\b',  # "finish installation"
            r'\btesting\b.*\bcomplete\b',  # "testing complete"
            r'\bverify\b.*\boperation\b',  # "verify operation"
            r'\binstallation\b.*\bcomplete\b',  # "installation complete"
        ]
        
        # Check if text contains completion indicators
        for pattern in completion_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True  # Text contains completion indicators
        
        # Check if text is longer than typical step and contains multiple steps
        step_count = len(re.findall(r'\b(?:Step|step)\s*\d+', text))
        if len(text) > self.target_chunk_size and step_count >= 3:
            return True  # Text contains multiple steps and is substantial
        
        return False  # Text is not a complete procedure
    
    def _contains_procedural_content(self, text: str) -> bool:
        """Check if text contains meaningful procedural content."""
        # Procedural content indicators
        procedural_indicators = [
            r'\b(?:step|Step|STEP)\s*\d+',  # Step numbers
            r'\b\d+\.\s+\w+',  # Numbered list items
            r'\b(?:install|Install|INSTALL)\b',  # Installation terms
            r'\b(?:connect|Connect|CONNECT)\b',  # Connection terms
            r'\b(?:remove|Remove|REMOVE)\b',  # Removal terms
            r'\b(?:place|Place|PLACE)\b',  # Placement terms
            r'\b(?:secure|Secure|SECURE)\b',  # Securing terms
            r'\b(?:verify|Verify|VERIFY)\b',  # Verification terms
            r'\b(?:warning|Warning|WARNING|caution|Caution|CAUTION)\b',  # Safety terms
        ]
        
        # Count procedural indicators
        indicator_count = 0  # Count of procedural indicators found
        for pattern in procedural_indicators:
            indicator_count += len(re.findall(pattern, text))
        
        # Text is procedural if it has indicators and meaningful length
        return indicator_count >= 2 and len(text.strip()) >= self.min_chunk_size // 2
    
    def _optimize_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """Optimize chunk sizes to meet constraints and improve retrieval."""
        optimized_chunks = []  # List to store optimized chunks
        
        i = 0  # Index for processing chunks
        while i < len(chunks):
            current_chunk = chunks[i]  # Get current chunk
            
            # Handle chunks that are too small
            if len(current_chunk) < self.min_chunk_size and i + 1 < len(chunks):
                # Try to merge with next chunk
                next_chunk = chunks[i + 1]  # Get next chunk
                merged_chunk = current_chunk + " " + next_chunk  # Merge chunks with space separator
                
                # If merged chunk is not too large, use it
                if len(merged_chunk) <= self.max_chunk_size:
                    optimized_chunks.append(merged_chunk)  # Add merged chunk
                    i += 2  # Skip both chunks as they are merged
                    continue  # Continue to next iteration
                else:
                    # If merging makes it too large, keep original if it's close to minimum
                    if len(current_chunk) >= self.min_chunk_size * 0.7:  # 70% of minimum is acceptable
                        optimized_chunks.append(current_chunk)  # Add current chunk
                    # Otherwise skip this too-small chunk
            
            # Handle chunks that are too large
            elif len(current_chunk) > self.max_chunk_size:
                # Split large chunk using fixed-length approach
                sub_chunks = chunk_by_fixed_length(current_chunk, chunk_size=self.target_chunk_size)  # Split large chunk
                optimized_chunks.extend(sub_chunks)  # Add all sub-chunks
            
            # Handle normal-sized chunks
            else:
                optimized_chunks.append(current_chunk)  # Add chunk as-is
            
            i += 1  # Move to next chunk
        
        # Log optimization results
        logger.debug(f"Chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks")
        
        return optimized_chunks  # Return optimized chunks


def chunk_adaptively(
    text: str,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    target_chunk_size: int = 800
) -> List[str]:
    """Convenience function for adaptive chunking.
    
    Parameters
    ----------
    text : str
        Input text to chunk
    min_chunk_size : int
        Minimum chunk size in characters
    max_chunk_size : int  
        Maximum chunk size in characters
    target_chunk_size : int
        Target chunk size in characters
        
    Returns
    -------
    List[str]
        List of adaptively chunked text
    """
    # Create adaptive chunker with specified parameters
    chunker = AdaptiveChunker(
        min_chunk_size=min_chunk_size,  # Set minimum chunk size
        max_chunk_size=max_chunk_size,  # Set maximum chunk size
        target_chunk_size=target_chunk_size  # Set target chunk size
    )
    
    # Apply adaptive chunking
    return chunker.chunk_text(text)  # Return chunked text


def get_chunker_info() -> Dict[str, Any]:
    """Get information about the adaptive chunker.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing chunker information and capabilities
    """
    return {
        'method': 'adaptive',  # Chunking method identifier
        'strategies': ['semantic', 'fixed_length', 'hybrid', 'procedural'],  # Available strategies
        'adapts_to_content': True,  # Whether chunker adapts to content type
        'supports_size_optimization': True,  # Whether chunker optimizes sizes
        'analyzes_document_type': True,  # Whether chunker detects document type
        'supports_procedural_content': True,  # Whether chunker handles step-by-step content
        'default_min_size': 100,  # Default minimum chunk size
        'default_max_size': 2000,  # Default maximum chunk size
        'default_target_size': 800,  # Default target chunk size
        'description': 'Adaptive chunker that selects optimal strategy based on content analysis, with specialized handling for installation and procedural documents'  # Description
    } 