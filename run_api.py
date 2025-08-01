#!/usr/bin/env python3
"""
Start the RAG Document Assistant Web Server

This script initializes and runs the Flask web application for the RAG system.
It provides a simple interface for document upload, processing, and querying.
"""

import os
import sys
import secrets
from app.api import app  # Import the Flask application

def main():
    """Start the RAG API server with helpful startup messages and API key info."""
    print("Starting RAG Document Assistant...")
    print("Web Interface: http://localhost:5000")
    print("Settings: http://localhost:5000/settings") 
    print("API Health Check: http://localhost:5000/api/health")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    # API Key setup
    api_key = os.environ.get('RAG_API_KEY')
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        os.environ['RAG_API_KEY'] = api_key
        print("[SECURITY] No RAG_API_KEY set. Generated random API key:")
        print(f"  X-API-Key: {api_key}")
        print("[SECURITY] To use your own key, set the RAG_API_KEY environment variable before starting the server.")
    else:
        print("[SECURITY] API key authentication is enabled.")
        print(f"  X-API-Key: {api_key}")

    if not os.path.exists('app'):
        print("Error: Please run this script from the rag-system directory")
        print("   Current directory:", os.getcwd())
        return 1
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"Server error: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 