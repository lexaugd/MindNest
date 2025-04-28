#!/usr/bin/env python
"""
Script to assess MindNest query responses for the enhanced
query classification and retrieval system.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assess_response(query, response, sources):
    """
    Assess the quality of a response.
    
    Args:
        query: The user query
        response: The system response
        sources: List of source documents
        
    Returns:
        Dictionary with assessment metrics
    """
    # Initialize assessment metrics
    assessment = {
        "query_type": detect_query_type(query),
        "response_length": len(response),
        "sources_count": len(sources) if sources else 0,
        "has_citation": sources is not None and len(sources) > 0,
        "relevance": "Unknown",  # Would need ground truth to assess
        "is_concise": len(response) < 500,
        "is_conversational": is_conversational_response(response),
    }
    
    # Analyze other aspects
    if assessment["query_type"] == "CONVERSATION":
        assessment["appropriate_response"] = assessment["is_conversational"]
    else:
        assessment["appropriate_response"] = assessment["has_citation"]
    
    return assessment

def detect_query_type(query):
    """
    Detect the type of query.
    
    Args:
        query: The user query
        
    Returns:
        Query type classification
    """
    query = query.lower().strip()
    
    # Simple classification logic
    if query.startswith("how are you") or query.startswith("hi") or query.startswith("hello"):
        return "CONVERSATION"
    elif query.startswith("what is") or query.startswith("tell me about"):
        return "DOCUMENT_QUERY"
    elif query.startswith("find") or query.startswith("search"):
        return "DOCUMENT_SEARCH"
    else:
        return "DOCUMENT_QUERY"  # Default

def is_conversational_response(response):
    """
    Check if a response is conversational in nature.
    
    Args:
        response: The system response
        
    Returns:
        Boolean indicating if response is conversational
    """
    conversational_patterns = [
        "I'm", "I am", "you", "your", "thanks", "thank you", 
        "hello", "hi", "hey", "good"
    ]
    
    lower_response = response.lower()
    return any(pattern in lower_response for pattern in conversational_patterns)

def print_assessment(query, response, sources, assessment):
    """
    Print the assessment in a readable format.
    
    Args:
        query: The user query
        response: The system response
        sources: List of source documents
        assessment: Assessment metrics dictionary
    """
    print("\n" + "="*80)
    print(f"QUERY: \"{query}\"")
    print(f"QUERY TYPE: {assessment['query_type']}")
    print("-"*80)
    print(f"RESPONSE: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
    print("-"*80)
    print(f"SOURCES: {len(sources) if sources else 0} document(s)")
    if sources and len(sources) > 0:
        for i, source in enumerate(sources[:3]):
            print(f"  {i+1}. {source}")
        if len(sources) > 3:
            print(f"  ... and {len(sources) - 3} more")
    print("-"*80)
    print("ASSESSMENT:")
    for key, value in assessment.items():
        if key not in ["query_type"]:  # Already printed above
            print(f"  {key}: {value}")
    print("="*80)

def main():
    """Main function to run the assessment."""
    # Sample data from the conversation
    samples = [
        {
            "query": "hi",
            "response": "The NBDIF consists of nine volumes, each addressing a specific key topic. Volumes 1-7 were conceptualized and written during Stage 1, while Volume 8 was added in Stage 2 to define general interfaces between the NBDRA components. Additionally, two new volumes, Volume 9 on adoption and modernization, and Volume 8 on general interfaces, were created during Stage 2. The finalized Version 2 documents can be downloaded from the V2.0 Final Version page of the NBD-PWG website.",
            "sources": [
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-component-specific-interface-requirements.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-introduction-secintroduction.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-background.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-scope-and-objectives-of-the-reference-architectures-subgroup.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-background.md"
            ]
        },
        {
            "query": "what is edi",
            "response": "Electronic Data Interchange (EDI) is the computer-to-computer exchange of business documents in a standard electronic format between business partners. EDI replaces paper-based documents such as purchase orders, invoices, and shipping notices with electronic equivalents.",
            "sources": [
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_basics.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_integration_patterns.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_integration_patterns.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_integration_patterns.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_integration_patterns.md"
            ]
        },
        {
            "query": "how are you",
            "response": "I'm doing well, thanks for asking. How about you?",
            "sources": [
                "/Users/alekseiugadov/Cursor/MindNest/docs/user_guides/usage.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/ai/usage_guide.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/edi/edi_introduction-component-specific-interface-requirements.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/query_classification_feedback.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/development/query_classification_feedback.md"
            ]
        },
        {
            "query": "what is object wrapper",
            "response": "Object Wrapper System provides a consistent interface for accessing different types of objects throughout the application. It allows for a uniform approach to data access regardless of the underlying data structure, simplifying business logic and making code more maintainable.",
            "sources": [
                "/Users/alekseiugadov/Cursor/MindNest/docs/md/03_Object_Wrappers.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/md/03_Object_Wrappers.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/md/03_Object_Wrappers.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/md/03_Object_Wrappers.md",
                "/Users/alekseiugadov/Cursor/MindNest/docs/md/03_Object_Wrappers.md"
            ]
        },
        {
            "query": "what is mindnest",
            "response": "MindNest is an intelligent documentation system that uses AI to understand and answer questions about your codebase. It leverages large language models (LLMs) to provide accurate, context-aware responses to questions about your documentation and code.",
            "sources": None  # Assuming no sources were provided for this response
        }
    ]
    
    # Run assessment on each sample
    print("\nASSESSING MINDNEST RESPONSES")
    print("=" * 80)
    
    overall_stats = {
        "total": len(samples),
        "appropriate_responses": 0,
        "with_citations": 0,
        "conversational_queries": 0,
        "document_queries": 0
    }
    
    for sample in samples:
        assessment = assess_response(
            sample["query"], 
            sample["response"], 
            sample["sources"]
        )
        
        print_assessment(
            sample["query"],
            sample["response"],
            sample["sources"],
            assessment
        )
        
        # Update overall stats
        if assessment["appropriate_response"]:
            overall_stats["appropriate_responses"] += 1
        
        if assessment["has_citation"]:
            overall_stats["with_citations"] += 1
            
        if assessment["query_type"] == "CONVERSATION":
            overall_stats["conversational_queries"] += 1
        else:
            overall_stats["document_queries"] += 1
    
    # Print overall assessment
    print("\nOVERALL ASSESSMENT")
    print("=" * 80)
    print(f"Total queries: {overall_stats['total']}")
    print(f"Appropriate responses: {overall_stats['appropriate_responses']} ({overall_stats['appropriate_responses']/overall_stats['total']*100:.1f}%)")
    print(f"Responses with citations: {overall_stats['with_citations']} ({overall_stats['with_citations']/overall_stats['total']*100:.1f}%)")
    print(f"Conversational queries: {overall_stats['conversational_queries']}")
    print(f"Document queries: {overall_stats['document_queries']}")
    print("=" * 80)

if __name__ == "__main__":
    main() 