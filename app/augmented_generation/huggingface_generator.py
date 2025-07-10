"""Hugging Face Transformers-based text generation for RAG systems.

This module provides free, local language model generation using Hugging Face's
transformers library. Models run entirely locally without API costs.

Popular free models:
- google/flan-t5-base: Good instruction following
- google/flan-t5-large: Better quality, larger size
- microsoft/DialoGPT-medium: Conversational model
"""

from typing import List, Dict, Any, Optional
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import required Hugging Face libraries
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    HF_AVAILABLE = True  # Flag to track if libraries are available
except ImportError:
    HF_AVAILABLE = False  # Set flag to False if imports fail
    logger.warning("Hugging Face transformers not available. Run: pip install transformers torch")


class HuggingFaceGenerator:
    """Local language model generator using Hugging Face Transformers.
    
    Completely free, runs locally, no API keys required.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base", 
        max_tokens: int = 400,
        temperature: float = 0.3,
        device: str = "auto"
    ):
        """Initialize Hugging Face generator.
        
        Args:
            model_name: Name of HF model (e.g., 'google/flan-t5-base')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        # Early error handling - check if HF is available
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers not installed. "
                "Run: pip install transformers torch"
            )
        
        self.model_name = model_name  # Store model name
        self.max_tokens = max_tokens  # Maximum response length
        self.temperature = temperature  # Control randomness
        
        # Determine device to use for inference
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer (lazy loading)
        self.tokenizer = None  # Will be loaded when first needed
        self.model = None  # Will be loaded when first needed
        self.pipeline = None  # Will be created when first needed
        
        logger.info(f"HuggingFaceGenerator initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")
    
    def _load_model(self) -> bool:
        """Load the model and tokenizer if not already loaded.
        
        Returns:
            bool: True if successful, False if failed
        """
        # Check if already loaded
        if self.model is not None and self.tokenizer is not None:
            return True  # Already loaded successfully
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer for the model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            
            # Load the model for sequence-to-sequence generation
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
            
            # Move model to appropriate device (GPU/CPU)
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
            
            # Create text generation pipeline for easier use
            self.pipeline = pipeline(
                "text2text-generation",  # Task type
                model=self.model,  # The loaded model
                tokenizer=self.tokenizer,  # The loaded tokenizer
                device=0 if self.device == "cuda" else -1,  # GPU index or CPU
                max_length=self.max_tokens,  # Maximum output length
                temperature=self.temperature,  # Sampling temperature
                do_sample=True if self.temperature > 0 else False  # Enable sampling if temp > 0
            )
            logger.info("Pipeline created successfully")
            
            return True  # Loading successful
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False  # Loading failed
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[str],
        max_context_length: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using retrieved chunks as context.
        
        Args:
            query: User's question/query
            retrieved_chunks: List of relevant text chunks from retrieval
            max_context_length: Maximum characters to include from chunks
            
        Returns:
            dict: Contains 'response', 'success', 'error' fields
        """
        # Early error handling - check if HF is available
        if not HF_AVAILABLE:
            return {
                'response': '',
                'success': False,
                'error': 'Hugging Face transformers not installed. Run: pip install transformers torch'
            }
        
        # Early validation - ensure we have chunks to work with
        if not retrieved_chunks:
            return {
                'response': '',
                'success': False,
                'error': 'No retrieved chunks provided for context.'
            }
        
        # Load model if not already loaded
        if not self._load_model():
            return {
                'response': '',
                'success': False,
                'error': f'Failed to load model: {self.model_name}'
            }
        
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(retrieved_chunks, max_context_length)
            
            # Create the prompt for the language model
            prompt = self._create_prompt(query, context)
            
            logger.info(f"Generating response for query: {query[:50]}...")
            
            # Generate response using the pipeline
            result = self.pipeline(
                prompt,  # Input prompt
                max_length=self.max_tokens,  # Maximum output length
                temperature=self.temperature,  # Sampling temperature
                pad_token_id=self.tokenizer.eos_token_id,  # Padding token
                num_return_sequences=1  # Generate single response
            )
            
            # Extract generated text from result
            generated_text = result[0]['generated_text'].strip()
            
            logger.info("Response generated successfully")
            
            # Return success response
            return {
                'response': generated_text,
                'success': True,
                'error': None,
                'model_used': self.model_name,
                'context_length': len(context),
                'device_used': self.device
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            # Return error response
            return {
                'response': '',
                'success': False,
                'error': f'Generation failed: {str(e)}'
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
            chunk_text = f"Source {i+1}: {chunk.strip()}"  # Format with source number
            
            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_text) > max_length:
                break  # Stop if we'd exceed the limit
            
            context_parts.append(chunk_text)  # Add chunk to context
            current_length += len(chunk_text)  # Update length counter
        
        # Join all context parts with newlines
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a focused prompt for technical documentation Q&A.
        
        Args:
            query: User's question
            context: Prepared context from retrieved chunks
            
        Returns:
            str: Complete prompt for the model
        """
        # Create direct, focused prompt optimized for FLAN-T5
        prompt = f"""Based on the following technical documentation, answer the question with specific technical details and step-by-step instructions when applicable.

Documentation:
{context}

Question: {query}

Technical Answer:"""
        
        return prompt  # Return the simplified technical prompt


class SmallLanguageModelGenerator:
    """Alternative generator using smaller, faster models for resource-constrained environments."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """Initialize with a smaller model for faster inference.
        
        Args:
            model_name: Name of small model to use
        """
        # Use base HuggingFaceGenerator with smaller model
        self.generator = HuggingFaceGenerator(
            model_name=model_name,
            max_tokens=150,  # Shorter responses for small models
            temperature=0.3,  # Lower temperature for more focused output
            device="cpu"  # Force CPU for better compatibility
        )
    
    def generate_response(self, query: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Generate response using the small model.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            dict: Generation result
        """
        # Delegate to the underlying generator
        return self.generator.generate_response(
            query, 
            retrieved_chunks, 
            max_context_length=500  # Smaller context for small models
        )


def create_huggingface_generator(
    model_name: str = "google/flan-t5-base",
    use_small_model: bool = False
) -> HuggingFaceGenerator:
    """Factory function to create a Hugging Face generator.
    
    Args:
        model_name: Name of the HF model to use
        use_small_model: Whether to use small/fast model variant
        
    Returns:
        HuggingFaceGenerator: Configured generator instance
    """
    if use_small_model:
        # Return small model generator for fast inference
        return SmallLanguageModelGenerator(model_name)
    else:
        # Return standard generator
        return HuggingFaceGenerator(model_name=model_name)


# Available free models for Hugging Face (no API keys required)
AVAILABLE_FREE_MODELS = {
    "google/flan-t5-small": "Small FLAN-T5 model - Fast, good for simple queries",
    "google/flan-t5-base": "Base FLAN-T5 model - Good balance of quality and speed",
    "google/flan-t5-large": "Large FLAN-T5 model - Higher quality, slower inference",
    "microsoft/DialoGPT-small": "Small conversational model - Good for chat",
    "microsoft/DialoGPT-medium": "Medium conversational model - Better responses",
    "distilgpt2": "Distilled GPT-2 - Fast text generation",
    "gpt2": "GPT-2 base model - Classic text generation"
}


def list_available_models() -> Dict[str, str]:
    """Get list of available free models for Hugging Face.
    
    Returns:
        dict: Model names mapped to descriptions
    """
    return AVAILABLE_FREE_MODELS  # Return available models dictionary


def check_huggingface_requirements() -> Dict[str, Any]:
    """Check if Hugging Face requirements are installed.
    
    Returns:
        dict: Status information about requirements
    """
    status = {
        'transformers_available': HF_AVAILABLE,
        'torch_available': False,
        'cuda_available': False,
        'recommended_install': 'pip install transformers torch'
    }
    
    # Check if torch is available and CUDA support
    if HF_AVAILABLE:
        try:
            import torch
            status['torch_available'] = True
            status['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            pass
    
    return status  # Return complete status information