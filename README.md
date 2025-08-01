# 🤖 Modular RAG System

A complete Retrieval-Augmented Generation (RAG) system with a modern web interface. This system allows you to ingest documents, ask questions, and get AI-powered answers based on your documents.

## Architecture

The system is designed with modularity in mind, allowing each component to be developed and modified independently:

```
rag-system/
├── app/
│   ├── chunkers/          # Text chunking strategies
│   ├── embedders/         # Text embedding models
│   ├── vector_db/         # Vector database operations
│   ├── retriever/         # Document retrieval logic
│   ├── augmented_generation/ # LLM generation with context
│   ├── data/             # Sample data and configuration
│   ├── main.py           # CLI application
│   └── api.py            # Web API
├── web_ui/               # Web user interface
│   ├── pages/            # HTML pages
│   └── assets/           # CSS, JS, images
├── requirements.txt      # Python dependencies
├── run_api.py           # Start web server
└── README.md            # This file
```

## Key Principles

- **Modularity**: Each component (chunkers, embedders, etc.) is independent
- **Simplicity**: Clean, focused implementations that are easy to understand
- **Extensibility**: Easy to add new algorithms and strategies
- **Interface Consistency**: Common patterns across all modules

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)
1. **Double-click `setup.bat`** (Windows) 
2. **Double-click `run.bat`** to start the system
3. **Open your browser** and go to: `http://localhost:8000`

### Option 2: PowerShell Users
1. **Right-click and "Run with PowerShell"**: `setup.ps1`
2. **Run**: `.\run.ps1` 
3. **Open your browser** and go to: `http://localhost:8000`

### Option 3: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 🎯 How to Use

1. **📝 Ingest Documents**: Paste text into the ingestion area and choose your chunking method
2. **❓ Ask Questions**: Type questions about your ingested documents  
3. **📊 System Status**: Check health and view stored chunks

## 📁 File Structure

The project now includes convenient setup scripts:

- `setup.bat` / `setup.ps1` - Set up virtual environment and install dependencies
- `run.bat` / `run.ps1` - Run the RAG system
- `static/` - Web interface files (HTML, CSS, JS separated properly)

## Components

- **Chunkers**: Break text into manageable pieces
- **Embedders**: Convert text to vector representations
- **Vector DB**: Store and search vector embeddings
- **Retriever**: Find relevant context for queries
- **Augmented Generation**: Generate responses using retrieved context

Each component follows the same pattern: simple functions with clear interfaces that can be easily swapped or extended. 

## API Key Security

**All sensitive API endpoints are protected by API key authentication.**

- On startup, if you do not set a `RAG_API_KEY` environment variable, the server will generate a secure random API key and print it to the console.
- To use your own key, set the environment variable before starting the server:
  - On Linux/macOS:
    ```sh
    export RAG_API_KEY="your-very-secret-key"
    python run_api.py
    ```
  - On Windows (PowerShell):
    ```powershell
    $env:RAG_API_KEY="your-very-secret-key"
    python run_api.py
    ```
- All clients must include this header in requests to protected endpoints:
  ```
  X-API-Key: your-very-secret-key
  ```
- Example with curl:
  ```sh
  curl -H "X-API-Key: your-very-secret-key" http://localhost:5000/api/documents/upload
  ```

**If you do not set a key, the server will print the generated key on startup. Use this key in your API requests.**

## Starting the Server

1. (Optional) Set your API key as described above.
2. Start the server:
   ```sh
   python run_api.py
   ```
3. The server will print the API key and usage instructions on startup.
4. Access the web interface at [http://localhost:5000](http://localhost:5000) 