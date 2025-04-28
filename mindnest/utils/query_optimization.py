"""
Query optimization utilities to improve response times.

This module provides functions for optimizing query processing,
including query categorization and response caching.
"""

import re
from functools import lru_cache
import hashlib

# Cached functions for improved performance

@lru_cache(maxsize=128)
def is_likely_document_query(query):
    """
    Determine if a query is likely document-related based on patterns.
    Uses caching to speed up repeated or similar queries.
    
    Args:
        query (str): The user's query text
        
    Returns:
        bool: True if the query is likely document-related, False otherwise
    """
    query_lower = query.lower()
    
    # Fast categorization patterns
    
    # Simple greetings and thank-yous (definitely not document queries)
    simple_patterns = [
        r"^hi+\s*$", r"^hello+\s*$", r"^hey+\s*$", r"^thanks+\s*$", 
        r"^thank you+\s*$", r"^ok+\s*$", r"^okay+\s*$", r"^good\s*$",
        r"^nice\s*$", r"^cool\s*$", r"^great\s*$", r"^bye+\s*$",
        r"^good morning\s*$", r"^good afternoon\s*$", r"^good evening\s*$",
        r"^how are you.*$", r"^how('s|\s+is) it going.*$", r"^how('s|\s+are) you doing.*$",
        r"^what('s|\s+is) up.*$", r"^sup+\s*$", r"^yo+\s*$", r"^wassup+\s*$",
        r"^what are you.*$", r"^who are you.*$", r"^tell me about yourself.*$"
    ]
    
    for pattern in simple_patterns:
        if re.match(pattern, query_lower):
            return False
    
    # Very short queries without document-related keywords are likely not document queries
    if len(query_lower.split()) < 3:
        document_keywords = ["doc", "file", "class", "method", "function", "code", 
                            "object", "system", "wrapper", "component"]
        
        if not any(keyword in query_lower for keyword in document_keywords):
            return False
    
    # If it starts with "find" or "search", it's definitely a document query
    if query_lower.startswith("find ") or query_lower.startswith("search "):
        return True
    
    # If it has a question mark, assume it's likely a document query
    if '?' in query_lower:
        return True
    
    # Check for question words with document-related keywords
    question_indicators = ["what", "how", "why", "when", "where", "which", "who", 
                          "explain", "describe", "tell me about", "show me", "what is", "what are"]
    
    # Simple conversation patterns (definitely not document queries)
    conversation_patterns = [
        r".*\b(how are you|how do you|how you doing|how is it going)\b.*",
        r".*\b(your name|who are you|tell me about yourself)\b.*",
        r".*\b(nice to meet you|pleasure to meet you)\b.*",
        r".*\b(what do you think|your opinion|you like|you enjoy)\b.*",
        r".*\b(weather|time is it|date today|current time)\b.*",
        r".*\b(good morning|good evening|good afternoon|good night)\b.*"
    ]
    
    for pattern in conversation_patterns:
        if re.search(pattern, query_lower):
            return False
    
    # If the query starts with a question indicator, assume it's a document query
    for indicator in question_indicators:
        if query_lower.startswith(indicator):
            return True
    
    # If it contains 5 or more words, it's likely a document query
    if len(query_lower.split()) >= 5:
        return True
    
    # Topic-related keywords (with very broad coverage)
    topic_keywords = ["object", "wrapper", "system", "class", "java", "code", 
                     "document", "file", "interface", "method", "function",
                     "application", "software", "program", "module", "component",
                     "architecture", "pattern", "design", "structure", "data",
                     "algorithm", "implementation", "api", "library", "framework",
                     "repository", "project", "source", "test", "example",
                     "feature", "integration", "service", "build", "deploy",
                     "config", "settings", "option", "parameter", "variable", 
                     "database", "storage", "memory", "cache", "file", "format", 
                     "access", "security", "validation", "process", "thread", "query"]
    
    has_question_word = any(word in query_lower for word in question_indicators)
    has_topic_keyword = any(word in query_lower for word in topic_keywords)
    
    if has_question_word or has_topic_keyword:
        return True
    
    # Likely document patterns (with broader patterns)
    document_patterns = [
        r".*\b(doc|document)(s|\w*)\b.*",
        r".*\b(code|coding|coded)\b.*",
        r".*\b(class|classes)\b.*",
        r".*\b(function|method|procedure)\b.*",
        r".*\b(implement|implementation)\b.*",
        r".*\b(api|interface|library)\b.*",
        r".*\b(system|framework|architecture)\b.*",
        r".*\b(object|instance|wrapper)\b.*",
        r".*\b(explain|describe|detail|summary)\b.*",
        r".*\b(definition|example|sample)\b.*",
        r".*\b(error|issue|problem|bug)\b.*",
        r".*\b(syntax|structure|format)\b.*",
        r".*\b(tool|utility|component)\b.*",
        r".*\b(file|directory|folder)\b.*",
        r".*\b(pattern|design)\b.*",
        r".*\b(build|compile|run)\b.*",
        r".*\b(test|debug|fix)\b.*"
    ]
    
    for pattern in document_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # By default, assume queries with 3+ words are likely legitimate document queries
    # This is a fallback to avoid false negatives
    if len(query_lower.split()) >= 3:
        return True
        
    # Default to false only for very short, unrecognized queries
    return False

@lru_cache(maxsize=128)
def categorize_query(query, model_size="large"):
    """
    Categorize a query with model-specific thresholds.
    
    Args:
        query (str): The user's query text
        model_size (str): "small" or "large" to adjust thresholds
        
    Returns:
        tuple: (query_type, processed_query)
    """
    query = query.strip()
    
    # Check if this is a search query (same for both models)
    if query.lower().startswith("find "):
        return "DOCUMENT_SEARCH", query[5:]
    
    # Model-specific adjustments for query categorization
    # For small models, be more aggressive about categorizing as CONCISE_QUERY
    # to avoid overwhelming the model
    if model_size == "small":
        # Additional concise patterns for small models
        simple_query_patterns = [
            r"^.{0,50}$",  # Very short queries
            r"^(what|how|why|when|where|which|who)\s+.{0,30}$",  # Simple question patterns
        ]
        
        for pattern in simple_query_patterns:
            if re.search(pattern, query.lower()):
                return "CONCISE_QUERY", query
    
    # Enhanced patterns for concise query detection
    concise_patterns = [
        # Explicit requests for brevity
        r"^(summarize|summarise|brief|concise|short|tl;dr|tldr|summary)\s+.+",
        r".+\s+in\s+(one|1|a|few)\s+(sentence|sentences|line|lines|paragraph|paragraphs|words).*$",
        r"^(explain|describe|define|what\s+is)\s+.+\s+(briefly|concisely|in short).*$",
        r"^(give|provide)\s+(me\s+)?(a\s+)?(brief|short|concise|quick|simple)\s+(summary|description|explanation|overview|answer)\s+.+$",
        r"^(can\s+you\s+)?(quickly|briefly|concisely)\s+(tell|explain|describe|summarize)\s+.+$",
        r"^short\s+(answer|explanation|description)\s+.+$",
        r".+\s+(keep\s+it|make\s+it)\s+(short|brief|concise|simple).*$",
        # Prefixes indicating desire for brevity
        r"^concise[:]?\s+.+$",
        r"^brief[:]?\s+.+$",
        r"^short[:]?\s+.+$",
        r"^simple[:]?\s+.+$",
        r"^tldr[:]?\s+.+$"
    ]
    
    for pattern in concise_patterns:
        if re.search(pattern, query.lower()):
            return "CONCISE_QUERY", query
    
    # Detect conversation patterns
    conversation_patterns = [
        r".*\b(how are you|how do you|how you doing|how is it going)\b.*",
        r".*\b(your name|who are you|tell me about yourself)\b.*",
        r".*\b(nice to meet you|pleasure to meet you)\b.*",
        r".*\b(what do you think|your opinion|you like|you enjoy)\b.*",
        r".*\b(weather|time is it|date today|current time)\b.*",
        r".*\b(good morning|good evening|good afternoon|good night)\b.*",
        r".*\b(hello|hi there|hey|greetings)\b.*",
        r".*\b(thanks|thank you|appreciate it)\b.*",
    ]
    
    # Calculate conversation likelihood based on patterns
    conversation_score = 0.0
    for pattern in conversation_patterns:
        if re.search(pattern, query.lower()):
            conversation_score = 0.9  # High confidence if matches a pattern
            break
    
    # If it's a very short query without question mark, likely conversation
    if len(query.split()) <= 2 and '?' not in query:
        conversation_score = max(conversation_score, 0.7)
    
    # Model-specific conversation thresholds
    if (model_size == "small" and conversation_score > 0.4) or (model_size == "large" and conversation_score > 0.7):
        return "CONVERSATION", query
            
    # Use is_likely_document_query for further classification
    if is_likely_document_query(query):
        # For small models, prefer concise queries for shorter questions
        if model_size == "small" and len(query.split()) < 8:
            return "CONCISE_QUERY", query
        return "DOCUMENT_QUERY", query
    
    # Otherwise, it's likely just conversational
    return "CONVERSATION", query

def create_query_hash(query):
    """
    Create a consistent hash for a query to use as a cache key.
    
    Args:
        query (str): The query to hash
        
    Returns:
        str: A hex digest of the hashed query
    """
    # Normalize query by lowercasing and removing extra whitespace
    normalized_query = re.sub(r'\s+', ' ', query.lower()).strip()
    
    # Create a hash of the normalized query
    return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()

def is_similar_query(query1, query2, threshold=0.8):
    """
    Check if two queries are similar based on word overlap.
    
    Args:
        query1 (str): First query
        query2 (str): Second query
        threshold (float): Similarity threshold, defaults to 0.8
        
    Returns:
        bool: True if queries are similar, False otherwise
    """
    # Normalize and tokenize queries
    words1 = set(re.sub(r'\W+', ' ', query1.lower()).split())
    words2 = set(re.sub(r'\W+', ' ', query2.lower()).split())
    
    # Remove common stopwords
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                "being", "and", "or", "but", "if", "then", "else", "when",
                "up", "down", "in", "out", "on", "off", "over", "under", "again",
                "further", "then", "once", "here", "there", "why", "how", "all",
                "any", "both", "each", "few", "more", "most", "other", "some",
                "such", "no", "not", "only", "own", "same", "so", "than", "too",
                "very", "s", "t", "can", "will", "just", "don", "should", "now",
                "what", "where", "which", "who", "whom", "this", "that", "these",
                "those", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
                "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "to", "from", "of"}
    
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return False
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold 

def process_query(text, model_capabilities=None):
    """
    Process the query and determine if it should be handled as a search or normal query.
    
    Args:
        text (str): The user's query text
        model_capabilities (dict, optional): Model capabilities for model-aware classification
        
    Returns:
        tuple: Query type and processed query
    """
    # Get model size for optimized classification
    model_size = "large"  # Default
    if model_capabilities and "model_size" in model_capabilities:
        model_size = model_capabilities["model_size"]
    
    try:
        # Use our classification function directly
        return categorize_query(text, model_size=model_size)
    except Exception as e:
        print(f"Error in query classification: {e}")
        print("Falling back to default classification")
        # Always default to document query on error
        return "DOCUMENT_QUERY", text 