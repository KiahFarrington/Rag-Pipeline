"""OpenAI-based response generator.

Uses OpenAI's API to generate responses based on query and retrieved context.
This module provides a simple interface for augmented generation.
"""

from typing import List, Optional
import os

# Global client instance (lazy loaded)
_client = None

def _get_client():
    """Lazy load the OpenAI client.
    
    Returns
    -------
    OpenAI
        The OpenAI client instance
    """
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required. "
                    "Set it with: export OPENAI_API_KEY=your_api_key"
                )
            _client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )
    return _client

def generate_response(query: str, context_chunks: List[str]) -> str:
    """Generate response using query and retrieved context.
    
    This function will be implemented step by step.
    """
    # Placeholder implementation
    return f"Generated response for: '{query}' (Generator not implemented yet)"

def _build_context_string(context_chunks: List[str]) -> str:
    """Build a formatted context string from chunks.
    
    Parameters
    ----------
    context_chunks : List[str]
        List of context chunks
        
    Returns
    -------
    str
        Formatted context string
    """
    if not context_chunks:
        return ""
    
    # Number the context chunks for clarity
    numbered_chunks = []
    for i, chunk in enumerate(context_chunks, 1):
        chunk = chunk.strip()
        if chunk:  # Only include non-empty chunks
            numbered_chunks.append(f"[{i}] {chunk}")
    
    return "\n\n".join(numbered_chunks)

def _build_prompt(query: str, context: str) -> str:
    """Build the complete prompt for the language model.
    
    Parameters
    ----------
    query : str
        User's query
    context : str
        Formatted context string
        
    Returns
    -------
    str
        Complete prompt for generation
    """
    prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can from the context.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt

def _generate_without_context(
    query: str, 
    model: str, 
    max_tokens: int, 
    temperature: float
) -> str:
    """Generate response without context (fallback).
    
    Parameters
    ----------
    query : str
        User's query
    model : str
        OpenAI model name
    max_tokens : int
        Maximum response tokens
    temperature : float
        Generation temperature
        
    Returns
    -------
    str
        Generated response
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. The user is asking a question but no specific context was provided."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"I apologize, but I don't have enough context to answer your question, and I encountered an error: {str(e)}"

def generate_simple_response(query: str, context_chunks: List[str]) -> str:
    """Generate a simple response with minimal configuration.
    
    This is a convenience function with sensible defaults.
    
    Parameters
    ----------
    query : str
        User's question
    context_chunks : List[str]
        Retrieved context chunks
        
    Returns
    -------
    str
        Generated response
    """
    return generate_response(
        query=query,
        context_chunks=context_chunks,
        model="gpt-3.5-turbo",
        max_tokens=300,
        temperature=0.3  # Lower temperature for more focused responses
    ) 