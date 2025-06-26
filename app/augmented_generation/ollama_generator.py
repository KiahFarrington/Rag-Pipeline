"""Ollama-based text generation for RAG systems.

This module provides free, local language model generation using Ollama.
Ollama allows you to run LLMs locally without API costs or internet dependency.

To use this:
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2` or `ollama pull mistral`
3. Run this generator
"""

import requests
import json
from typing import List, Dict, Any, Optional


class OllamaGenerator:
    """Local language model generator using Ollama.
    
    Completely free, runs locally, no API keys required.
    """
    
    def __init__(
        self, 
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """Initialize Ollama generator.
        
        Args:
            model_name: Name of Ollama model to use (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server URL (default localhost)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name  # Store the model name to use
        self.base_url = base_url  # Ollama server endpoint
        self.max_tokens = max_tokens  # Limit response length
        self.temperature = temperature  # Control randomness in generation
        self.api_url = f"{base_url}/api/generate"  # Full API endpoint
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is running and accessible.
        
        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            # Test connection to Ollama server
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200  # Return True if server responds
        except requests.exceptions.RequestException:
            return False  # Return False if connection fails
    
    def _check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            # Get list of available models from Ollama
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()  # Parse JSON response
                # Extract model names from the response
                available_models = [model['name'].split(':')[0] for model in models_data.get('models', [])]
                return self.model_name in available_models  # Check if our model is in the list
            return False  # Return False if request failed
        except requests.exceptions.RequestException:
            return False  # Return False if connection fails
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[str],
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """Generate response using retrieved chunks as context.
        
        Args:
            query: User's question/query
            retrieved_chunks: List of relevant text chunks from retrieval
            max_context_length: Maximum characters to include from chunks
            
        Returns:
            dict: Contains 'response', 'success', 'error' fields
        """
        # Early error handling - check Ollama connection
        if not self._check_ollama_connection():
            return {
                'response': '',
                'success': False,
                'error': 'Ollama server not running. Please start Ollama and try again.'
            }
        
        # Early error handling - check model availability
        if not self._check_model_availability():
            return {
                'response': '',
                'success': False,
                'error': f'Model "{self.model_name}" not found. Please run: ollama pull {self.model_name}'
            }
        
        # Early validation - ensure we have chunks to work with
        if not retrieved_chunks:
            return {
                'response': '',
                'success': False,
                'error': 'No retrieved chunks provided for context.'
            }
        
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(retrieved_chunks, max_context_length)
            
            # Create the prompt for the language model
            prompt = self._create_prompt(query, context)
            
            # Prepare request payload for Ollama API
            payload = {
                "model": self.model_name,  # Which model to use
                "prompt": prompt,  # The input prompt
                "stream": False,  # Get complete response, not streaming
                "options": {
                    "num_predict": self.max_tokens,  # Maximum tokens to generate
                    "temperature": self.temperature,  # Sampling temperature
                    "top_p": 0.9,  # Top-p sampling for diversity
                    "stop": ["Human:", "Assistant:", "\n\nQuestion:"]  # Stop sequences
                }
            }
            
            # Send request to Ollama API
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30 second timeout for generation
            )
            
            # Handle successful response
            if response.status_code == 200:
                result = response.json()  # Parse JSON response
                generated_text = result.get('response', '').strip()  # Extract generated text
                
                # Return success response
                return {
                    'response': generated_text,
                    'success': True,
                    'error': None,
                    'model_used': self.model_name,
                    'context_length': len(context)
                }
            else:
                # Handle API error response
                return {
                    'response': '',
                    'success': False,
                    'error': f'Ollama API error: {response.status_code} - {response.text}'
                }
                
        except requests.exceptions.Timeout:
            # Handle timeout error
            return {
                'response': '',
                'success': False,
                'error': 'Request timeout. The model might be taking too long to respond.'
            }
        except requests.exceptions.RequestException as e:
            # Handle other request errors
            return {
                'response': '',
                'success': False,
                'error': f'Request failed: {str(e)}'
            }
        except Exception as e:
            # Handle unexpected errors
            return {
                'response': '',
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def _prepare_context(self, chunks: List[str], max_length: int) -> str:
        """Prepare context string from retrieved chunks.
        
        Args:
            chunks: List of text chunks
            max_length: Maximum total character length
            
        Returns:
            str: Formatted context string
        """
        context_parts = []  # List to accumulate context pieces
        current_length = 0  # Track total character count
        
        # Add chunks until we hit the length limit
        for i, chunk in enumerate(chunks):
            chunk_text = f"Source {i+1}: {chunk.strip()}"  # Format chunk with source number
            
            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_text) > max_length:
                break  # Stop adding chunks if we'd exceed limit
            
            context_parts.append(chunk_text)  # Add chunk to context
            current_length += len(chunk_text)  # Update length counter
        
        # Join all context parts with double newlines
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt for the language model.
        
        Args:
            query: User's question
            context: Prepared context from retrieved chunks
            
        Returns:
            str: Complete prompt for the model
        """
        # Create a structured prompt that guides the model
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using only the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but thorough in your response
- Cite specific sources when possible (e.g., "According to Source 1...")

Answer:"""
        
        return prompt  # Return the complete prompt


def create_ollama_generator(model_name: str = "llama2") -> OllamaGenerator:
    """Factory function to create an Ollama generator.
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        OllamaGenerator: Configured generator instance
    """
    # Create and return generator with specified model
    return OllamaGenerator(model_name=model_name)


# Available free models for Ollama (no API keys required)
AVAILABLE_FREE_MODELS = {
    "llama2": "Meta's Llama 2 - General purpose, good balance of quality and speed",
    "mistral": "Mistral 7B - Fast and efficient, good for most tasks", 
    "codellama": "Code-focused Llama model - Best for code-related queries",
    "llama2:13b": "Larger Llama 2 model - Higher quality but slower",
    "neural-chat": "Intel's neural chat model - Optimized for conversation",
    "starling-lm": "Starling language model - Good general performance"
}


def list_available_models() -> Dict[str, str]:
    """Get list of available free models for Ollama.
    
    Returns:
        dict: Model names mapped to descriptions
    """
    return AVAILABLE_FREE_MODELS  # Return the available models dictionary