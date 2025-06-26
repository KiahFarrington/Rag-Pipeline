"""Test file for all chunkers in the chunkers package."""

import os
from .fixed_length_chunker import chunk_by_fixed_length
from .semantic_chunker import chunk_by_semantics


def load_sample_data():
    """Load sample text from the data folder."""
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'sample.txt')
    
    # Read the sample text
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        # Fallback to simple text if file not found
        return "Simple test text for chunking."


def test_all_chunkers():
    """Test both chunkers with sample data."""
    # Load sample text from data folder
    sample_text = load_sample_data()
    
    chunks_fixed = chunk_by_fixed_length(sample_text)  # Test fixed-length chunker
    chunks_semantic = chunk_by_semantics(sample_text)  # Test semantic chunker
    
    # Simple assertions to verify chunkers work
    assert len(chunks_fixed) > 0, "Fixed chunker should return chunks"
    assert len(chunks_semantic) > 0, "Semantic chunker should return chunks"
    assert isinstance(chunks_fixed, list), "Fixed chunker should return list"
    assert isinstance(chunks_semantic, list), "Semantic chunker should return list"
    
    # Test empty input
    assert chunk_by_fixed_length("") == [], "Should handle empty input"
    assert chunk_by_semantics("") == [], "Should handle empty input"
    
    # Test single word
    single_word = "Hello"
    assert chunk_by_fixed_length(single_word) == [single_word], "Should handle single word"
    assert chunk_by_semantics(single_word) == [single_word], "Should handle single word"
    
    return chunks_fixed, chunks_semantic


def run_chunker_tests():
    """Run all chunker tests and return results."""
    try:
        # Run the tests
        fixed_chunks, semantic_chunks = test_all_chunkers()
        
        # Return test results
        return {
            'fixed_chunks_count': len(fixed_chunks),
            'semantic_chunks_count': len(semantic_chunks),
            'tests_passed': True,
            'fixed_chunks': fixed_chunks,
            'semantic_chunks': semantic_chunks
        }
    except AssertionError as e:
        # Return failed test info
        return {
            'tests_passed': False,
            'error': str(e)
        }
    except Exception as e:
        # Return unexpected error info
        return {
            'tests_passed': False,
            'error': f"Unexpected error: {str(e)}"
        } 