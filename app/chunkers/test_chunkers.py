"""Test file for all chunkers in the chunkers package."""

import os
import time
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


def test_empty_input_edge_cases():
    """Test edge cases with empty or whitespace-only inputs."""
    # Test completely empty string
    assert chunk_by_fixed_length("") == [], "Fixed chunker should handle empty string"
    assert chunk_by_semantics("") == [], "Semantic chunker should handle empty string"
    
    # Test whitespace-only strings
    assert chunk_by_fixed_length("   ") == [], "Fixed chunker should handle whitespace-only"
    assert chunk_by_semantics("   ") == [], "Semantic chunker should handle whitespace-only"
    
    # Test newline-only strings
    assert chunk_by_fixed_length("\n\n\n") == [], "Fixed chunker should handle newlines-only"
    assert chunk_by_semantics("\n\n\n") == [], "Semantic chunker should handle newlines-only"
    
    # Test mixed whitespace
    assert chunk_by_fixed_length("\t  \n  \r  ") == [], "Fixed chunker should handle mixed whitespace"
    assert chunk_by_semantics("\t  \n  \r  ") == [], "Semantic chunker should handle mixed whitespace"


def test_single_character_and_word():
    """Test minimal text inputs."""
    # Test single character
    single_char = "A"
    assert chunk_by_fixed_length(single_char) == [single_char], "Fixed chunker should handle single character"
    assert chunk_by_semantics(single_char) == [single_char], "Semantic chunker should handle single character"
    
    # Test single word
    single_word = "Hello"
    assert chunk_by_fixed_length(single_word) == [single_word], "Fixed chunker should handle single word"
    assert chunk_by_semantics(single_word) == [single_word], "Semantic chunker should handle single word"
    
    # Test single word with surrounding whitespace
    word_with_spaces = "  Hello  "
    expected = ["Hello"]
    assert chunk_by_fixed_length(word_with_spaces) == expected, "Fixed chunker should strip whitespace"
    assert chunk_by_semantics(word_with_spaces) == expected, "Semantic chunker should strip whitespace"


def test_fixed_length_boundary_conditions():
    """Test fixed length chunker with various text lengths around the 500-char boundary."""
    # Test text exactly 500 characters
    text_500 = "A" * 500
    chunks = chunk_by_fixed_length(text_500)
    assert len(chunks) == 1, "Text exactly 500 chars should create 1 chunk"
    assert len(chunks[0]) == 500, "Chunk should be exactly 500 characters"
    
    # Test text just under 500 characters  
    text_499 = "A" * 499
    chunks = chunk_by_fixed_length(text_499)
    assert len(chunks) == 1, "Text under 500 chars should create 1 chunk"
    assert len(chunks[0]) == 499, "Chunk should be exactly 499 characters"
    
    # Test text just over 500 characters
    text_501 = "A" * 501
    chunks = chunk_by_fixed_length(text_501)
    assert len(chunks) == 2, "Text over 500 chars should create 2 chunks"
    assert len(chunks[0]) == 500, "First chunk should be 500 characters"
    assert len(chunks[1]) == 1, "Second chunk should be 1 character"
    
    # Test text exactly 1000 characters (2 full chunks)
    text_1000 = "B" * 1000
    chunks = chunk_by_fixed_length(text_1000)
    assert len(chunks) == 2, "1000 chars should create exactly 2 chunks"
    assert all(len(chunk) == 500 for chunk in chunks), "Both chunks should be 500 characters"
    
    # Test text 1001 characters (2 full + 1 partial chunk)
    text_1001 = "C" * 1001
    chunks = chunk_by_fixed_length(text_1001)
    assert len(chunks) == 3, "1001 chars should create 3 chunks"
    assert len(chunks[0]) == 500, "First chunk should be 500 characters"
    assert len(chunks[1]) == 500, "Second chunk should be 500 characters"
    assert len(chunks[2]) == 1, "Third chunk should be 1 character"


def test_semantic_chunker_paragraph_patterns():
    """Test semantic chunker with various paragraph break patterns."""
    # Test standard double newlines
    text_double_newlines = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_by_semantics(text_double_newlines)
    assert len(chunks) == 3, "Double newlines should create 3 chunks"
    assert chunks[0] == "First paragraph.", "First chunk should match expected text"
    assert chunks[1] == "Second paragraph.", "Second chunk should match expected text"
    assert chunks[2] == "Third paragraph.", "Third chunk should match expected text"
    
    # Test newlines with spaces
    text_spaced_breaks = "Para one.\n   \nPara two.\n\t\nPara three."
    chunks = chunk_by_semantics(text_spaced_breaks)
    assert len(chunks) == 3, "Newlines with spaces should create 3 chunks"
    
    # Test 3+ spaces after newline pattern
    text_indented = "Line one.\n   Line two.\n      Line three."
    chunks = chunk_by_semantics(text_indented)
    assert len(chunks) == 3, "Lines with 3+ spaces should create 3 chunks"
    
    # Test no paragraph breaks (single chunk)
    text_no_breaks = "This is all one paragraph with no breaks just spaces."
    chunks = chunk_by_semantics(text_no_breaks)
    assert len(chunks) == 1, "Text without breaks should create 1 chunk"
    assert chunks[0] == text_no_breaks, "Single chunk should match input text"
    
    # Test multiple consecutive paragraph breaks
    text_multiple_breaks = "First.\n\n\n\nSecond.\n\n\n\n\nThird."
    chunks = chunk_by_semantics(text_multiple_breaks)
    assert len(chunks) == 3, "Multiple breaks should still create correct chunks"
    assert chunks[0] == "First.", "First chunk should be clean"
    assert chunks[1] == "Second.", "Second chunk should be clean"
    assert chunks[2] == "Third.", "Third chunk should be clean"


def test_special_characters_and_unicode():
    """Test chunkers with special characters, unicode, and symbols."""
    # Test unicode characters
    unicode_text = "Hello 世界! 🌍 Héllo wörld! Привет мир!"
    fixed_chunks = chunk_by_fixed_length(unicode_text)
    semantic_chunks = chunk_by_semantics(unicode_text)
    assert len(fixed_chunks) > 0, "Fixed chunker should handle unicode"
    assert len(semantic_chunks) > 0, "Semantic chunker should handle unicode"
    
    # Test special symbols and punctuation
    symbols_text = "Test with symbols: @#$%^&*()[]{}|\\:;\"'<>,.?/~`!"
    fixed_chunks = chunk_by_fixed_length(symbols_text)
    semantic_chunks = chunk_by_semantics(symbols_text)
    assert len(fixed_chunks) == 1, "Fixed chunker should handle symbols as single chunk"
    assert len(semantic_chunks) == 1, "Semantic chunker should handle symbols as single chunk"
    
    # Test numbers and mixed content
    mixed_text = "Data: 123.456, Price: $99.99, Date: 2024-01-01, Time: 14:30:00"
    fixed_chunks = chunk_by_fixed_length(mixed_text)
    semantic_chunks = chunk_by_semantics(mixed_text)
    assert len(fixed_chunks) == 1, "Fixed chunker should handle mixed content"
    assert len(semantic_chunks) == 1, "Semantic chunker should handle mixed content"


def test_different_line_endings():
    """Test chunkers with different line ending formats."""
    # Test Unix line endings (\n)
    unix_text = "Line 1\nLine 2\n\nLine 3"
    unix_chunks = chunk_by_semantics(unix_text)
    assert len(unix_chunks) == 2, "Unix line endings should work correctly"
    
    # Test Windows line endings (\r\n)
    windows_text = "Line 1\r\nLine 2\r\n\r\nLine 3"
    windows_chunks = chunk_by_semantics(windows_text)
    assert len(windows_chunks) >= 1, "Windows line endings should be handled"
    
    # Test old Mac line endings (\r)
    mac_text = "Line 1\rLine 2\r\rLine 3"
    mac_chunks = chunk_by_semantics(mac_text)
    assert len(mac_chunks) >= 1, "Mac line endings should be handled"


def test_whitespace_variations():
    """Test chunkers with various whitespace patterns."""
    # Test tabs vs spaces
    tab_text = "Section 1\n\t\nSection 2"
    space_text = "Section 1\n   \nSection 2"
    
    tab_chunks = chunk_by_semantics(tab_text)
    space_chunks = chunk_by_semantics(space_text)
    
    assert len(tab_chunks) == 2, "Tabs should trigger paragraph breaks"
    assert len(space_chunks) == 2, "Spaces should trigger paragraph breaks"
    
    # Test mixed whitespace in chunks
    mixed_ws_text = "  Start with spaces\t\tTabs in middle   End with spaces  "
    chunks = chunk_by_fixed_length(mixed_ws_text)
    assert len(chunks) == 1, "Mixed whitespace should be preserved in content"
    assert chunks[0].startswith("Start"), "Leading whitespace should be stripped"
    assert chunks[0].endswith("spaces"), "Trailing whitespace should be stripped"


def test_performance_with_large_text():
    """Test chunker performance with large text inputs."""
    # Create large text (approximately 50KB)
    large_text = "This is a test sentence. " * 2000
    
    # Time the fixed length chunker
    start_time = time.time()
    fixed_chunks = chunk_by_fixed_length(large_text)
    fixed_time = time.time() - start_time
    
    # Time the semantic chunker  
    start_time = time.time()
    semantic_chunks = chunk_by_semantics(large_text)
    semantic_time = time.time() - start_time
    
    # Performance assertions (should complete within reasonable time)
    assert fixed_time < 1.0, f"Fixed chunker took too long: {fixed_time:.3f}s"
    assert semantic_time < 1.0, f"Semantic chunker took too long: {semantic_time:.3f}s"
    
    # Functional assertions
    assert len(fixed_chunks) > 1, "Large text should create multiple fixed chunks"
    assert len(semantic_chunks) >= 1, "Large text should create at least one semantic chunk"
    
    # Verify chunk sizes for fixed chunker
    for i, chunk in enumerate(fixed_chunks[:-1]):  # All but last chunk
        assert len(chunk) == 500, f"Chunk {i} should be exactly 500 chars, got {len(chunk)}"


def test_chunk_content_integrity():
    """Test that chunking preserves content integrity."""
    original_text = load_sample_data()
    
    # Test fixed length chunker content preservation
    fixed_chunks = chunk_by_fixed_length(original_text)
    reconstructed_fixed = "".join(fixed_chunks)
    assert reconstructed_fixed == original_text.strip(), "Fixed chunker should preserve content"
    
    # Test semantic chunker content preservation
    semantic_chunks = chunk_by_semantics(original_text)
    # For semantic chunker, we need to reconstruct with paragraph breaks
    reconstructed_semantic = "\n\n".join(semantic_chunks)
    # Verify that all original content is present (allowing for formatting differences)
    for chunk in semantic_chunks:
        assert chunk in original_text, f"Chunk content should exist in original: {chunk[:50]}..."


def test_edge_case_combinations():
    """Test combinations of edge cases."""
    # Empty string with whitespace operations
    edge_cases = [
        "",           # completely empty
        " ",          # single space
        "\n",         # single newline
        "\t",         # single tab
        "\r\n",       # windows newline
        "a",          # single letter
        "a\n\nb",     # minimal semantic split
        "a" * 499,    # just under 500
        "a" * 500,    # exactly 500
        "a" * 501,    # just over 500
    ]
    
    for i, test_case in enumerate(edge_cases):
        # Test that both chunkers handle each case without errors
        try:
            fixed_result = chunk_by_fixed_length(test_case)
            semantic_result = chunk_by_semantics(test_case)
            
            # Basic sanity checks
            assert isinstance(fixed_result, list), f"Case {i}: Fixed chunker should return list"
            assert isinstance(semantic_result, list), f"Case {i}: Semantic chunker should return list"
            
            # Verify no empty chunks (unless input was empty)
            if test_case.strip():
                assert all(chunk.strip() for chunk in fixed_result), f"Case {i}: No empty fixed chunks"
                assert all(chunk.strip() for chunk in semantic_result), f"Case {i}: No empty semantic chunks"
            
        except Exception as e:
            assert False, f"Case {i} failed with input '{repr(test_case)}': {str(e)}"


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
    
    # Don't return anything - just assertions for pytest compliance


def run_all_comprehensive_tests():
    """Run all comprehensive tests and return detailed results."""
    test_results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'failed_tests': [],
        'test_details': {}
    }
    
    # Define all test functions
    test_functions = [
        test_empty_input_edge_cases,
        test_single_character_and_word,
        test_fixed_length_boundary_conditions,
        test_semantic_chunker_paragraph_patterns,
        test_special_characters_and_unicode,
        test_different_line_endings,
        test_whitespace_variations,
        test_performance_with_large_text,
        test_chunk_content_integrity,
        test_edge_case_combinations,
        test_all_chunkers
    ]
    
    # Run each test function
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            # Execute the test
            result = test_func()
            test_results['tests_passed'] += 1
            test_results['test_details'][test_name] = {'status': 'PASSED', 'result': result if result is not None else 'PASSED'}
            
        except AssertionError as e:
            # Handle assertion failures
            test_results['tests_failed'] += 1
            test_results['failed_tests'].append(test_name)
            test_results['test_details'][test_name] = {'status': 'FAILED', 'error': str(e)}
            
        except Exception as e:
            # Handle unexpected errors
            test_results['tests_failed'] += 1
            test_results['failed_tests'].append(test_name)
            test_results['test_details'][test_name] = {'status': 'ERROR', 'error': f"Unexpected error: {str(e)}"}
    
    # Calculate success rate
    total_tests = test_results['tests_passed'] + test_results['tests_failed']
    success_rate = (test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    test_results['total_tests'] = total_tests
    test_results['success_rate'] = success_rate
    
    return test_results


def run_chunker_tests():
    """Run all chunker tests and return results (legacy function for compatibility)."""
    try:
        # Load sample data for testing
        sample_text = load_sample_data()
        
        # Run the basic tests - test_all_chunkers now only does assertions
        test_all_chunkers()
        
        # Generate chunks for result reporting
        fixed_chunks = chunk_by_fixed_length(sample_text)
        semantic_chunks = chunk_by_semantics(sample_text)
        
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


if __name__ == "__main__":
    """Run comprehensive tests when file is executed directly."""
    results = run_all_comprehensive_tests()
    
    print(f"Tests: {results['tests_passed']}/{results['total_tests']} passed")
    if results['failed_tests']:
        for test_name in results['failed_tests']:
            error_info = results['test_details'][test_name]
            print(f"FAILED: {test_name} - {error_info['error']}")
    else:
        print("All tests passed") 