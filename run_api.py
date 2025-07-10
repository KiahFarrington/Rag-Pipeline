#!/usr/bin/env python3
"""Simple script to run the RAG API server.

This script starts the Flask development server for the RAG system.
Users can then access the web interface at http://localhost:5000
"""

import os  # Operating system interface
import sys  # System-specific parameters and functions
from app.api import app  # Import the Flask application

def main():
    """Start the RAG API server with helpful startup messages."""
    
    print("🚀 Starting RAG Document Assistant...")  # Startup message
    print("=" * 50)  # Visual separator
    
    # Check if we're in the right directory
    if not os.path.exists('app'):
        print("❌ Error: Please run this script from the rag-system directory")  # Error if wrong directory
        print("   Current directory:", os.getcwd())  # Show current directory
        return 1  # Exit with error code
    
    # Print access information
    print("🌐 Web Interface:")  # Web interface header
    print("   Main Page:    http://localhost:5000")  # Main page URL
    print("   Settings:     http://localhost:5000/settings")  # Settings page URL
    print()  # Empty line for spacing
    
    print("📋 API Endpoints:")  # API endpoints header
    print("   Health Check: http://localhost:5000/api/health")  # Health check endpoint
    print("   Upload Doc:   POST /api/documents/ingest")  # Document upload endpoint
    print("   Query:        POST /api/query")  # Query endpoint
    print()  # Empty line for spacing
    
    print("💡 Quick Start:")  # Quick start instructions
    print("   1. Open http://localhost:5000 in your browser")  # Step 1
    print("   2. Paste some text in the upload area")  # Step 2
    print("   3. Click 'Process Document'")  # Step 3
    print("   4. Ask questions about your document!")  # Step 4
    print()  # Empty line for spacing
    
    print("⚙️  Configure system behavior at the Settings page")  # Settings info
    print("🛑 Press Ctrl+C to stop the server")  # Stop instructions
    print("=" * 50)  # Visual separator
    
    try:
        # Start the Flask development server
        app.run(
            debug=True,  # Enable debug mode for development
            host='0.0.0.0',  # Listen on all interfaces
            port=5000,  # Use port 5000
            use_reloader=False  # Disable reloader to avoid duplicate startup messages
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")  # Graceful shutdown message
        return 0  # Exit successfully
    except Exception as e:
        print(f"\n❌ Server error: {e}")  # Error message
        return 1  # Exit with error code

if __name__ == '__main__':
    """Entry point when script is run directly."""
    exit_code = main()  # Run main function and get exit code
    sys.exit(exit_code)  # Exit with the returned code 