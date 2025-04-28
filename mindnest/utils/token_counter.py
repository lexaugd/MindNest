"""
Token counting utilities for accurate context window management.

This module provides functions to count tokens for different model types,
ensuring we respect context window limits accurately.
"""

import re
from typing import Optional, Dict, Any

# Default encoding for most LLaMA-based models
DEFAULT_ENCODING = "cl100k_base"  # Works well for most modern models

def get_tokenizer(model_encoding: Optional[str] = None):
    """
    Get the appropriate tokenizer based on model encoding.
    
    Args:
        model_encoding (str, optional): The encoding to use. Defaults to cl100k_base.
        
    Returns:
        tiktoken.Encoding: The tokenizer
    """
    try:
        import tiktoken
        encoding_name = model_encoding or DEFAULT_ENCODING
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Warning: Error getting tokenizer: {e}")
        # Fallback to approximate token counting if tiktoken fails
        return None

def count_tokens(text: str, model_encoding: Optional[str] = None) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text (str): The text to count tokens for
        model_encoding (str, optional): The encoding to use. Defaults to cl100k_base.
        
    Returns:
        int: The number of tokens
    """
    tokenizer = get_tokenizer(model_encoding)
    
    if tokenizer:
        # Use tiktoken for accurate counting
        return len(tokenizer.encode(text))
    else:
        # Fallback to approximate counting if tokenizer isn't available
        return approximate_token_count(text)

def approximate_token_count(text: str) -> int:
    """
    Approximates token count when tokenizer is unavailable.
    
    This is a fallback method that uses simple rules to estimate tokens.
    For English text, ~4 characters ≈ 1 token, with variations for punctuation.
    
    Args:
        text (str): The text to count tokens for
        
    Returns:
        int: Estimated token count
    """
    # Clean the text - remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Count words (tokens are roughly aligned with words for English)
    words = text.split()
    num_words = len(words)
    
    # Count characters (excluding spaces)
    num_chars = len(text) - text.count(' ')
    
    # Special tokens for punctuation, etc.
    special_chars = sum(1 for c in text if c in ".,;:!?()[]{}-+*/='\"")
    
    # Estimate: blend of word count and character count approaches
    # Words are typically 1-2 tokens, some punctuation gets separate tokens
    estimate = num_words + (special_chars * 0.5) + (num_chars / 4) * 0.2
    
    # Return conservative estimate rounded up
    return int(estimate) + 1

def get_max_tokens_for_model(model_capabilities: Dict[str, Any]) -> int:
    """
    Get the maximum tokens allowed in the context window for a model.
    
    Args:
        model_capabilities (Dict[str, Any]): The model capabilities dictionary
        
    Returns:
        int: Maximum tokens allowed
    """
    # Extract context window from capabilities
    context_window = model_capabilities.get("context_window", 2048)
    
    # For small models, reserve more space for the response
    if model_capabilities.get("model_size", "large") == "small":
        # Reserve 25% of context window for response
        return int(context_window * 0.75)
    else:
        # Reserve 15% of context window for response
        return int(context_window * 0.85)

def truncate_to_token_limit(text: str, max_tokens: int, model_encoding: Optional[str] = None) -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text (str): Text to truncate
        max_tokens (int): Maximum tokens allowed
        model_encoding (str, optional): Encoding to use
        
    Returns:
        str: Truncated text that fits within token limit
    """
    tokenizer = get_tokenizer(model_encoding)
    
    if not tokenizer:
        # Fallback to character-based truncation if tokenizer is unavailable
        # Approximately 4 characters per token for English text
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text
        return text[:char_limit - 3] + "..."
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # If already under the limit, return the original text
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    # Add ellipsis to indicate truncation
    return truncated_text + "..." 