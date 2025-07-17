#!/usr/bin/env python3
"""
MCP Server Startup Script

This script starts the Model Context Protocol (MCP) server for the RAG system.
The MCP server exposes RAG functionality as standardized tools that can be used
by AI assistants and other MCP-compatible clients.

Usage:
    python run_mcp.py [transport]

Arguments:
    transport: The transport method to use (stdio, sse, streamable-http)
               Defaults to stdio for maximum compatibility

Examples:
    python run_mcp.py                    # Uses stdio transport (default)
    python run_mcp.py stdio              # Explicitly use stdio
    python run_mcp.py sse                # Use Server-Sent Events transport
    python run_mcp.py streamable-http    # Use Streamable HTTP transport
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the app directory to the Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from app.mcp.server import get_mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the MCP server."""
    
    # Get transport method from command line arguments
    transport = "stdio"  # Default transport
    if len(sys.argv) > 1:
        transport = sys.argv[1].lower()
    
    # Validate transport method
    valid_transports = ["stdio", "sse", "streamable-http"]
    if transport not in valid_transports:
        logger.error(f"Invalid transport '{transport}'. Valid options: {', '.join(valid_transports)}")
        sys.exit(1)
    
    logger.info(f"Starting RAG System MCP Server with transport: {transport}")
    logger.info("The MCP server exposes the following capabilities:")
    logger.info("  Tools: upload_document, ingest_text, query_knowledge, get_system_status")
    logger.info("  Resources: rag://system/status")
    logger.info("  Prompts: create_knowledge_query, create_document_summary_request")
    
    try:
        # Get the MCP server instance
        mcp_server = get_mcp_server()
        
        # Run the server with the specified transport
        if transport == "stdio":
            logger.info("Running MCP server with stdio transport (for CLI clients and testing)")
            mcp_server.run(transport="stdio")
        elif transport == "sse":
            logger.info("Running MCP server with SSE transport on http://localhost:8000")
            mcp_server.run(transport="sse", host="0.0.0.0", port=8000)
        elif transport == "streamable-http":
            logger.info("Running MCP server with Streamable HTTP transport on http://localhost:8000")
            mcp_server.run(transport="streamable-http", host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        logger.info("MCP server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)
    
    logger.info("MCP server stopped")

if __name__ == "__main__":
    main() 