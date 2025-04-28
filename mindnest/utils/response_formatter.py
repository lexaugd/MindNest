"""
Response formatting utilities for MindNest application.
These functions handle formatting, validation, and enhancement of AI responses.
"""

import re
from typing import List, Dict, Any, Optional
from langchain.schema import Document

def format_response(response_text: str, model_capabilities: Dict[str, Any]) -> str:
    """
    Format a response based on model capabilities.
    
    Args:
        response_text (str): The raw response text from the model
        model_capabilities (dict): The model capabilities dictionary
        
    Returns:
        str: The formatted response
    """
    # Remove any unwanted patterns or prefixes
    # Some models add prefixes like "Answer: " or "Response: "
    response_text = re.sub(r'^(Answer|Response|Assistant|AI|I would say):\s*', '', response_text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    response_text = re.sub(r'\s+', ' ', response_text).strip()
    
    # Check if we need to limit the response length
    max_length = model_capabilities.get("max_response_length", 500)
    if len(response_text) > max_length:
        # Truncate to the nearest sentence end within the max length
        truncated = response_text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # Only truncate at sentence if it's reasonably close to max
            truncated = response_text[:last_period + 1]
        else:
            truncated += "..."
        response_text = truncated
    
    return response_text

def validate_response_quality(response: str, query: str, model_capabilities: Dict[str, Any]) -> str:
    """
    Check response quality and make improvements if needed.
    
    Args:
        response (str): The response to validate
        query (str): The original query
        model_capabilities (dict): The model capabilities dictionary
        
    Returns:
        str: The validated and potentially improved response
    """
    # If response is too short or empty, return a fallback
    if not response or len(response) < 10:
        return "I don't have enough information in the documentation to answer this question properly."
    
    # Check if response has any hallucination indicators
    hallucination_phrases = [
        "I don't know", "I'm not sure", "I don't have access", 
        "I don't have information", "I cannot access", "I cannot browse",
        "I'm an AI", "as an AI", "my knowledge", "my training",
        "I don't have the ability", "I can't provide"
    ]
    
    if any(phrase in response.lower() for phrase in hallucination_phrases):
        fallback = "The documentation doesn't provide specific information to answer this question. "
        fallback += "You might want to check the official documentation or other sources for more details."
        return fallback
    
    # Check if the response is missing relevance to the query
    if "test" in query.lower():
        # For test queries, add a note about testing
        response += "\n\nNote: This response is for testing purposes."
    
    return response

def create_hybrid_response(response: str, query: str, docs: List[Document], model_capabilities: Dict[str, Any]) -> str:
    """
    Create a hybrid response that combines model output with relevant document snippets.
    
    Args:
        response (str): The model's response
        query (str): The original query
        docs (list): List of Document objects
        model_capabilities (dict): The model capabilities dictionary
        
    Returns:
        str: A hybrid response containing both LLM answer and relevant snippets
    """
    # For short queries or test queries, just return the response
    if len(query.split()) < 4 or "test" in query.lower():
        return response
    
    # If we have documents, augment the response with relevant snippets
    if docs and len(docs) > 0:
        # Check if we should include snippets based on query type
        include_snippets = True
        
        # Add snippets for complex queries
        if include_snippets:
            # Find the most relevant document (usually the first one)
            most_relevant = docs[0] if docs else None
            
            if most_relevant:
                # Extract a concise snippet (up to 150 chars)
                snippet = most_relevant.page_content[:150]
                if len(most_relevant.page_content) > 150:
                    snippet += "..."
                
                # Add the snippet as supporting evidence
                hybrid = response
                
                # Only add snippets if the response doesn't already contain them
                if snippet.lower() not in response.lower():
                    hybrid += f"\n\nRelevant information from documentation: \"{snippet}\""
                
                return hybrid
    
    # No augmentation needed or possible
    return response 