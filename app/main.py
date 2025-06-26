"""Main entry point for the Modular RAG System.

This is a step-by-step built RAG system - no web framework, just pure Python.
Perfect for understanding each component as we build it.
"""

from chunkers.test_chunkers import run_chunker_tests


def main():
    """Main function to demonstrate the RAG pipeline."""
    # Run chunker tests
    test_results = run_chunker_tests()
    
    # Return results for potential use
    return test_results


if __name__ == "__main__":
    main() 