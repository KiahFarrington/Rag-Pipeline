#!/usr/bin/env python3
"""
Start the RAG Document Assistant Web Server

This script initializes and runs the Flask web application for the RAG system.
It provides a simple interface for document upload, processing, and querying.
"""

import os  # Operating system interface
import sys  # System-specific parameters and functions
from app.api import app  # Import the Flask application

def main():
    """Start the RAG API server with helpful startup messages."""
    
    print("Starting RAG Document Assistant...")
    print("Web Interface: http://localhost:5000")
    print("Settings: http://localhost:5000/settings") 
    print("API Health Check: http://localhost:5000/api/health")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app'):
        print("Error: Please run this script from the rag-system directory")
        return 1  # Exit with error code
    
    try:
        # Start the Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")  # Graceful shutdown message
        return 0  # Exit successfully
    except Exception as e:
        print(f"Server error: {e}")
        return 1  # Exit with error code

if __name__ == '__main__':
    """Entry point when script is run directly."""
    exit_code = main()  # Run main function and get exit code
    sys.exit(exit_code)  # Exit with the returned code 