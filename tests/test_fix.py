#!/usr/bin/env python3
"""Test script to verify RAG system fixes work correctly."""

import requests
import json
import time


def test_health_endpoint():
    """Test the health endpoint to check system status."""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✓ Health endpoint working")
            print(f"  Status: {health_data.get('status', 'unknown')}")
            
            # Check embedding test results
            embedding_test = health_data.get('diagnostics', {}).get('embedding_test', {})
            if embedding_test.get('status') == 'success':
                print(f"✓ Embedding system working ({embedding_test.get('method')})")
            else:
                print(f"⚠ Embedding system: {embedding_test.get('status', 'unknown')}")
                if 'error' in embedding_test:
                    print(f"  Error: {embedding_test['error']}")
            
            return True
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Health endpoint error: {str(e)}")
        return False


def test_simple_upload():
    """Test uploading a simple text document."""
    try:
        # Create a simple test text file
        test_content = """This is a test document for the RAG system.
        
It contains multiple paragraphs to test the chunking functionality.

The system should be able to process this text and create embeddings successfully.
        
This test verifies that our fixes for the SentenceTransformer device issues work correctly."""
        
        # Prepare the file upload
        files = {
            'file': ('test_document.txt', test_content, 'text/plain')
        }
        
        # Send upload request
        response = requests.post('http://localhost:5000/api/documents/upload', files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Document upload successful")
            print(f"  Document ID: {result.get('document_id')}")
            print(f"  Chunks created: {result.get('chunks_created')}")
            print(f"  Config used: {result.get('config_used', {}).get('embedding_method')}")
            return True
        else:
            print(f"✗ Document upload failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Upload test error: {str(e)}")
        return False


def test_query():
    """Test querying the uploaded document."""
    try:
        query_data = {
            'query': 'What is this document about?',
            'top_k': 3,
            'use_generation': False
        }
        
        response = requests.post('http://localhost:5000/api/query', 
                               json=query_data, 
                               headers={'Content-Type': 'application/json'},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            chunks = result.get('retrieved_chunks', [])
            print("✓ Query successful")
            print(f"  Retrieved {len(chunks)} chunks")
            if chunks:
                print(f"  First chunk preview: {chunks[0]['text'][:100]}...")
            return True
        else:
            print(f"✗ Query failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Query test error: {str(e)}")
        return False


def main():
    """Run all tests to verify the RAG system is working."""
    print("🧪 Testing RAG System Fixes...")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health_endpoint():
        tests_passed += 1
    
    if test_simple_upload():
        tests_passed += 1
    
    if test_query():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("System ready at: http://localhost:5000")
    else:
        print("Some tests failed. The system may have reduced functionality.")


if __name__ == '__main__':
    main() 