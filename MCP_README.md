# RAG System MCP Server

## What is MCP?

The **Model Context Protocol (MCP)** is an open standard developed by Anthropic that enables AI assistants to connect to external tools, systems, and data sources in a standardized way. Think of it as a universal connector that lets AI assistants use your RAG system without needing custom integrations.

## What does this MCP Server do?

This MCP server acts as a **bridge** between AI assistants (like Claude Desktop) and your RAG system. It exposes your RAG functionality as standardized "tools" that AI assistants can discover and use automatically.

### Key Benefits:
- **No Code Changes**: Your existing Flask API keeps working exactly as before
- **Universal Compatibility**: Works with any MCP-compatible AI assistant
- **Standardized Interface**: Uses the MCP protocol for consistent interaction
- **Enhanced AI Integration**: Shows how your RAG system can integrate with AI assistants

## Available Tools

The MCP server exposes these tools to AI assistants:

### 📄 Document Management
- **`upload_document`**: Upload and process files (PDF, TXT, etc.)
- **`ingest_text`**: Add text content directly to the knowledge base

### 🔍 Knowledge Querying  
- **`query_knowledge`**: Ask questions and get AI-generated answers from your documents
- **`get_system_status`**: Check if the RAG system is healthy and running

### 📋 Resources
- **`rag://system/status`**: Real-time system status information

### 💬 Prompts
- **`create_knowledge_query`**: Generate effective prompts for querying knowledge
- **`create_document_summary_request`**: Create prompts for document summarization

## How to Use

### 1. Start Your RAG System
First, make sure your Flask API is running:
```bash
python run_api.py
```

### 2. Start the MCP Server
```bash
# Basic usage (stdio transport)
python run_mcp.py

# Or specify transport method
python run_mcp.py stdio              # For CLI clients
python run_mcp.py sse                # For web-based clients  
python run_mcp.py streamable-http    # For modern HTTP clients
```

### 3. Connect an AI Assistant

#### For Claude Desktop:
Add this to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "rag-system": {
      "command": "python",
      "args": ["run_mcp.py"],
      "cwd": "/path/to/your/rag-system"
    }
  }
}
```

#### For Testing:
Use the MCP Inspector (if installed):
```bash
npx @modelcontextprotocol/inspector python run_mcp.py
```

## Example Usage

Once connected, you can ask the AI assistant to:

- **"Upload this PDF and process it into the knowledge base"**
- **"What does the document say about machine learning?"**  
- **"Check if the RAG system is running properly"**
- **"Summarize the main topics from all uploaded documents"**

The AI assistant will automatically use the appropriate MCP tools to fulfill these requests.

## Technical Details

### Architecture
```
AI Assistant (Claude, etc.)
       ↓ (MCP Protocol)
MCP Server (run_mcp.py)  
       ↓ (HTTP calls)
Flask RAG API (run_api.py)
       ↓
Your RAG System
```

### File Structure
```
rag-system/
├── app/mcp/
│   ├── __init__.py
│   └── server.py          # Main MCP server implementation
├── run_mcp.py            # MCP server startup script
└── MCP_README.md         # This documentation
```

### Configuration
- **API Base URL**: `http://localhost:5000/api` (configurable in `app/mcp/server.py`)
- **Default Transport**: `stdio`
- **Default Timeout**: 30 seconds

## Troubleshooting

### Common Issues

1. **"Connection refused"**: Make sure your Flask API is running first
2. **"Module not found"**: Ensure the MCP package is installed (`pip install mcp`)
3. **"Transport error"**: Try different transport methods (stdio is most reliable)

### Logs
The MCP server provides detailed logging to help diagnose issues:
```bash
python run_mcp.py  # Check console output for errors
```

## What This Demonstrates

This MCP server implementation showcases:

1. **Modern AI Integration**: How RAG systems can integrate with AI assistants using standard protocols
2. **Protocol Compliance**: Proper implementation of the MCP specification
3. **Practical Architecture**: Clean separation between the RAG system and AI integration layer
4. **Extensibility**: How to expose complex functionality through simple, standardized interfaces

## Future Possibilities

With MCP, your RAG system could be:
- Integrated into multiple AI assistants simultaneously
- Combined with other MCP servers for complex workflows
- Used by AI agents for autonomous research and analysis
- Extended with additional tools for advanced RAG operations

This demonstrates the potential for building sophisticated, interoperable AI systems using standardized protocols. 