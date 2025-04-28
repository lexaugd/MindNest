"""
Response quality evaluation utilities for MindNest.

This module provides tools for evaluating and ensuring the quality of AI-generated
responses, including relevance scoring, hallucination detection, and formatting checks.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_response_quality(
    query: str,
    response: str,
    context_docs: List[Any],
    model_capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the quality of an AI-generated response.
    
    Args:
        query: The user's original query
        response: The AI-generated response text
        context_docs: The documents used as context for generating the response
        model_capabilities: Dictionary of model capabilities including size
        
    Returns:
        Dictionary containing quality metrics and evaluation results
    """
    # Initialize evaluation metrics
    evaluation = {
        "relevance_score": 0.0,
        "potential_hallucination": False,
        "formatting_issues": [],
        "length_appropriate": True,
        "question_answered": False,
        "factual_accuracy": None,  # Would require verification against ground truth
        "overall_quality": 0.0
    }
    
    # Evaluate relevance to the query
    evaluation["relevance_score"] = evaluate_relevance(query, response)
    
    # Check for potential hallucinations
    evaluation["potential_hallucination"] = detect_hallucination(response, context_docs)
    
    # Check formatting quality
    evaluation["formatting_issues"] = check_formatting(response)
    
    # Check if length is appropriate for query complexity
    evaluation["length_appropriate"] = is_length_appropriate(query, response, model_capabilities)
    
    # Check if the question was directly answered
    evaluation["question_answered"] = check_question_answered(query, response)
    
    # Calculate overall quality score (weighted average of other metrics)
    evaluation["overall_quality"] = calculate_overall_quality(evaluation)
    
    return evaluation

def evaluate_relevance(query: str, response: str) -> float:
    """
    Evaluate how relevant the response is to the original query.
    
    Args:
        query: The user's original query
        response: The AI-generated response
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    # Basic implementation - check for query terms in response
    query_terms = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    response_terms = set(re.sub(r'[^\w\s]', '', response.lower()).split())
    
    # Calculate overlap percentage
    if not query_terms:
        return 0.5  # Default for empty queries
    
    overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
    
    # Apply simple modifier for response length appropriateness
    words_in_response = len(response.split())
    length_factor = 1.0
    
    if words_in_response < 5:
        length_factor = 0.5  # Penalize extremely short responses
    elif words_in_response > 500:
        length_factor = 0.8  # Slightly penalize extremely verbose responses
        
    return min(1.0, overlap * length_factor * 1.5)  # Scale up but cap at 1.0

def detect_hallucination(response: str, context_docs: List[Any]) -> bool:
    """
    Detect potential hallucinations in the response by checking if key statements
    are supported by the context documents.
    
    Args:
        response: The AI-generated response
        context_docs: List of context documents used for generation
        
    Returns:
        Boolean indicating whether potential hallucinations were detected
    """
    # Basic implementation - check for confidence phrases without support
    hallucination_signals = [
        "I am certain that",
        "It is definitely",
        "The document explicitly states",
        "The exact date is",
        "The precise number is",
        "The author specifically mentioned"
    ]
    
    # Combine all context text
    context_text = " ".join([doc.page_content for doc in context_docs if hasattr(doc, 'page_content')])
    context_text = context_text.lower()
    
    # Check for definitive statements not supported by context
    for signal in hallucination_signals:
        if signal.lower() in response.lower():
            # Get the sentence containing the signal
            pattern = re.escape(signal.lower()) + r'[^.!?]*[.!?]'
            matches = re.findall(pattern, response.lower())
            
            for match in matches:
                # Extract key terms from the statement
                key_terms = [term for term in re.sub(r'[^\w\s]', '', match).split() 
                             if len(term) > 4 and term not in ["about", "these", "those", "their"]]
                
                # Check if at least half of the key terms appear in context
                terms_found = sum(1 for term in key_terms if term in context_text)
                if len(key_terms) > 0 and terms_found / len(key_terms) < 0.5:
                    return True
    
    return False

def check_formatting(response: str) -> List[str]:
    """
    Check the response for formatting issues.
    
    Args:
        response: The AI-generated response
        
    Returns:
        List of formatting issues found
    """
    issues = []
    
    # Check for incomplete sentences
    if re.search(r'[a-zA-Z],\s*$', response) or response.endswith('...'):
        issues.append("Response ends with incomplete sentence")
    
    # Check for overuse of bullet points
    bullet_points = len(re.findall(r'[\n\r]\s*[-*•]', response))
    if bullet_points > 10:
        issues.append(f"Excessive use of bullet points ({bullet_points} found)")
    
    # Check for formatting consistency in lists
    if bullet_points > 0:
        bullet_styles = set(re.findall(r'[\n\r]\s*([-*•])', response))
        if len(bullet_styles) > 1:
            issues.append("Inconsistent bullet point styles")
    
    # Check for code block formatting - improved regex to catch more cases
    code_blocks = len(re.findall(r'```[\w]*', response))
    closing_code_blocks = len(re.findall(r'```\s*$|```\s*[\n\r]', response))
    if code_blocks != closing_code_blocks:
        issues.append("Mismatched code block formatting")
    
    # Check for reasonable paragraph length
    paragraphs = re.split(r'[\n\r]{2,}', response)
    for i, para in enumerate(paragraphs):
        if len(para.split()) > 150:
            issues.append(f"Paragraph {i+1} is excessively long ({len(para.split())} words)")
    
    return issues

def is_length_appropriate(query: str, response: str, model_capabilities: Dict[str, Any]) -> bool:
    """
    Determine if the response length is appropriate for the query complexity.
    
    Args:
        query: The user's original query
        response: The AI-generated response
        model_capabilities: Dictionary of model capabilities
        
    Returns:
        Boolean indicating whether the length is appropriate
    """
    query_words = len(query.split())
    response_words = len(response.split())
    
    # For simple questions (less than 15 words), responses should be concise
    if query_words < 15:
        if response_words > 200:
            return False
    
    # For medium complexity questions
    elif query_words < 30:
        if response_words > 400:
            return False
    
    # For very complex questions or small models, allow more verbose responses
    elif model_capabilities.get("size") == "small":
        if response_words > 800:
            return False
    
    # For large models, expect more concise responses
    elif model_capabilities.get("size") == "large":
        if response_words > 600:
            return False
    
    # Check if response is too short
    if response_words < 5:
        return False
        
    return True

def check_question_answered(query: str, response: str) -> bool:
    """
    Check if the response directly addresses the question asked.
    
    Args:
        query: The user's original query
        response: The AI-generated response
        
    Returns:
        Boolean indicating whether the question appears to be answered
    """
    # Extract question words and key terms from query
    question_words = ["what", "when", "where", "who", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does"]
    
    # For non-questions like "Tell me about X", check if the response contains the subject
    if query.lower().startswith("tell me about") or query.lower().startswith("explain") or query.lower().startswith("describe"):
        subject = re.sub(r'tell me about|explain|describe', '', query.lower(), flags=re.IGNORECASE).strip().strip('.')
        # If the subject appears in the response, consider it answered
        if subject and subject.lower() in response.lower():
            return True
    
    # Check if query is a question
    is_question = any(query.lower().startswith(qw) for qw in question_words) or "?" in query
    
    if not is_question:
        # If not a question and it has a reasonable response with more than 10 words
        return len(response.split()) >= 10
    
    # For questions, look for semantic indicators that the question was addressed
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Extract main question subject (naive approach)
    query_nouns = []
    for word in re.sub(r'[^\w\s]', '', query_lower).split():
        if len(word) > 3 and word not in question_words:
            query_nouns.append(word)
    
    # Check if response contains the main subject terms
    if query_nouns:
        subject_presence = sum(1 for noun in query_nouns if noun in response_lower)
        if subject_presence / len(query_nouns) >= 0.5:
            return True  # If half or more of the nouns are present, likely answered
    
    # Check for direct answer patterns - expanded patterns
    if "?" in query:
        answer_patterns = [
            r'^\s*yes\b',
            r'^\s*no\b',
            r'\bthe answer is\b',
            r'\bto answer your question\b',
            r'^\s*it is\b',
            r'^\s*they are\b',
            r'^\s*there are\b',
            r'\bwas created\b',
            r'\bwas developed\b',
            r'\bwas founded\b',
            r'\bwas established\b',
            r'\bin \d{4}\b'  # Years like "in 1991"
        ]
        if any(re.search(pattern, response_lower) for pattern in answer_patterns):
            return True
    
    # Special handling for specific question types
    if "when" in query_lower or "date" in query_lower or "year" in query_lower:
        # Look for date-like patterns in response
        date_patterns = [
            r'\b\d{4}\b',  # Years like 1991
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b'
        ]
        if any(re.search(pattern, response_lower, re.IGNORECASE) for pattern in date_patterns):
            return True
    
    # Default: assume the question is answered if we have a substantial response
    return len(response.split()) >= 20

def calculate_overall_quality(evaluation: Dict[str, Any]) -> float:
    """
    Calculate an overall quality score based on individual metrics.
    
    Args:
        evaluation: Dictionary containing individual quality metrics
        
    Returns:
        Overall quality score between 0.0 and 1.0
    """
    # Define weights for each factor
    weights = {
        "relevance_score": 0.4,
        "potential_hallucination": 0.3,
        "formatting_issues": 0.1,
        "length_appropriate": 0.1,
        "question_answered": 0.1
    }
    
    # Calculate weighted score
    score = 0.0
    
    # Add relevance score (already between 0-1)
    score += evaluation["relevance_score"] * weights["relevance_score"]
    
    # Subtract for hallucinations
    if evaluation["potential_hallucination"]:
        score -= weights["potential_hallucination"]
    else:
        score += weights["potential_hallucination"]
    
    # Subtract for formatting issues (normalized to 0-1 range)
    formatting_penalty = min(1.0, len(evaluation["formatting_issues"]) / 5)
    score += (1.0 - formatting_penalty) * weights["formatting_issues"]
    
    # Add for appropriate length
    if evaluation["length_appropriate"]:
        score += weights["length_appropriate"]
    
    # Add for answering the question
    if evaluation["question_answered"]:
        score += weights["question_answered"]
    
    # Ensure score is within 0-1 range
    return max(0.0, min(1.0, score))

def should_regenerate_response(evaluation: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if a response should be regenerated based on quality evaluation.
    
    Args:
        evaluation: Dictionary containing quality evaluation metrics
        
    Returns:
        Tuple of (bool, str) indicating whether to regenerate and the reason
    """
    regenerate = False
    reason = ""
    
    # Check for critical issues
    if evaluation["potential_hallucination"]:
        regenerate = True
        reason = "Potential hallucination detected"
    elif evaluation["overall_quality"] < 0.5:
        regenerate = True
        reason = f"Low quality score ({evaluation['overall_quality']:.2f})"
    elif not evaluation["question_answered"]:
        regenerate = True
        reason = "Question not directly answered"
    elif len(evaluation["formatting_issues"]) >= 3:
        regenerate = True
        reason = f"Multiple formatting issues: {', '.join(evaluation['formatting_issues'][:3])}"
    
    return regenerate, reason 