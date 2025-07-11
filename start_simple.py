#!/usr/bin/env python3
"""
Simple RAG Server Start - Uses TF-IDF to avoid PyTorch issues
"""

import os
import sys

# Set environment variable to use TF-IDF by default
os.environ['RAG_DEFAULT_EMBEDDING'] = 'tfidf'

def main():
    print("🚀 Starting RAG System (Simple Mode)")
    print("📍 Web Interface: http://localhost:5000")
    print("⚙️  Using TF-IDF embeddings (no PyTorch required)")
    print("=" * 50)
    
    try:
        # Import and modify the app to use TF-IDF
        from app.api import app, rag_state
        
        # Configure for TF-IDF to avoid PyTorch
        rag_state.config['embedding_method'] = 'tfidf'
        rag_state.config['generation_method'] = 'none'  # Disable generation for now
        
        print("✅ Configuration set to TF-IDF mode")
        print("🌐 Starting server on http://localhost:5000")
        
        # Start the server
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 