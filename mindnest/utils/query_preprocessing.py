"""
Query preprocessing module for MindNest.

This module provides functions for preprocessing queries
before they are sent to the vector store or LLM, including:
- Neural query rewriting
- Entity extraction
- Query expansion
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

class QueryPreprocessor:
    """
    Class for preprocessing queries to improve retrieval quality.
    Handles query rewriting, expansion, and entity extraction
    using dynamic learning-based approaches.
    """
    
    def __init__(self, llm=None, embedding_model=None):
        """
        Initialize query preprocessor.
        
        Args:
            llm: Language model for query rewriting (optional)
            embedding_model: Embedding model for semantic operations (optional)
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.enable_neural_rewriting = llm is not None
        
        # Track successful and unsuccessful queries for learning
        self.query_history = {
            "successful": [],  # Queries that led to good responses
            "unsuccessful": []  # Queries that needed improvement
        }
        
        # Most common entities in documents (to be populated by document analysis)
        self.document_entities = {}
        
    def analyze_documents(self, documents):
        """
        Analyze documents to extract common entities and terminology.
        This builds a domain-specific vocabulary dynamically rather than hardcoding.
        
        Args:
            documents: List of documents to analyze
        """
        if not documents:
            return
            
        # This would use the embedding model to identify key terms
        # by analyzing their importance in the document collection
        logger.info(f"Analyzing {len(documents)} documents to extract domain terminology")
        
        # This implementation would be completed with actual analysis
        # that extracts terminology from documents without hardcoding
        
    def learn_from_feedback(self, original_query, response, was_successful):
        """
        Learn from user feedback about query effectiveness.
        
        Args:
            original_query: The original query
            response: The response provided
            was_successful: Whether the response was helpful
        """
        # Store the query in appropriate history category
        if was_successful:
            self.query_history["successful"].append({
                "query": original_query,
                "response": response
            })
        else:
            self.query_history["unsuccessful"].append({
                "query": original_query,
                "response": response
            })
            
        # Limit history size
        max_history = 100
        if len(self.query_history["successful"]) > max_history:
            self.query_history["successful"] = self.query_history["successful"][-max_history:]
        if len(self.query_history["unsuccessful"]) > max_history:
            self.query_history["unsuccessful"] = self.query_history["unsuccessful"][-max_history:]
            
        logger.debug(f"Updated query history: {len(self.query_history['successful'])} successful, {len(self.query_history['unsuccessful'])} unsuccessful")
    
    def preprocess_query(self, query: str, query_type: str = None) -> Dict[str, Any]:
        """
        Preprocess a query to improve retrieval quality.
        
        Args:
            query: The original user query
            query_type: The type of query (if known)
            
        Returns:
            Dict containing processed query information:
                - original_query: Original query text
                - rewritten_query: Rewritten query for better retrieval
                - expanded_queries: List of query variations for better recall
                - query_type: Classified query type
        """
        logger.debug(f"Preprocessing query: {query}")
        
        # Initialize result with original query
        result = {
            "original_query": query,
            "rewritten_query": query,
            "expanded_queries": [],
            "query_type": query_type
        }
        
        # Rewrite query using LLM if enabled
        if self.enable_neural_rewriting and self.llm:
            rewritten_query = self._rewrite_query_with_llm(query)
            result["rewritten_query"] = rewritten_query
            logger.debug(f"Rewritten query: {rewritten_query}")
            
            # Generate query expansions
            expanded_queries = self._expand_query(query, rewritten_query)
            result["expanded_queries"] = expanded_queries
            logger.debug(f"Generated {len(expanded_queries)} query expansions")
        
        return result
    
    def _rewrite_query_with_llm(self, query: str) -> str:
        """
        Rewrite a query using LLM to improve retrieval quality.
        Uses examples from successful queries when available.
        
        Args:
            query: The original query
            
        Returns:
            Rewritten query optimized for retrieval
        """
        if not self.llm:
            return query
            
        try:
            # Default to original query if LLM fails
            rewritten_query = query
            
            # Build dynamic examples from history
            examples = ""
            if self.query_history["successful"]:
                # Use successful queries as examples
                examples = "Here are examples of queries that worked well:\n"
                for i, item in enumerate(self.query_history["successful"][-3:]):  # Use last 3
                    examples += f"Example {i+1}: \"{item['query']}\"\n"
                examples += "\n"
            
            # Prompt for query rewriting
            prompt = f"""
            You are an AI assistant helping to improve search queries for a documentation system.
            Rewrite the following query to make it more effective for retrieving relevant documentation.
            
            {examples}
            
            Original query: "{query}"
            
            Rewritten query (provide ONLY the rewritten query, nothing else):
            """
            
            # Get response from LLM
            response = self.llm(prompt)
            
            # Clean up response
            if response:
                # Remove quotes if present
                rewritten_query = response.strip().strip('"\'')
                
                # If rewritten query is too different, fall back to original
                if len(rewritten_query) < 3:
                    logger.warning(f"Rewritten query too short, falling back to original: {query}")
                    return query
                    
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query with LLM: {e}")
            return query
    
    def _expand_query(self, original_query: str, rewritten_query: str) -> List[str]:
        """
        Generate expanded variations of the query for better recall.
        
        Args:
            original_query: The original user query
            rewritten_query: The rewritten query
            
        Returns:
            List of query variations
        """
        expanded_queries = []
        
        # Add original if different from rewritten
        if original_query != rewritten_query:
            expanded_queries.append(original_query)
            
        # If we have an LLM, generate more variations
        if self.llm:
            try:
                # Dynamic example generation
                examples = ""
                if self.query_history["successful"]:
                    # Use successful queries as examples for expansion
                    success_query = self.query_history["successful"][-1]["query"]
                    examples = f"For example, the query \"{success_query}\" could have alternatives like:\n"
                    examples += "- Alternative version focusing on key concepts\n"
                    examples += "- Alternative version with different terminology\n\n"
                
                # Prompt for query expansion
                prompt = f"""
                Generate 2-3 alternative search queries for retrieving documentation based on this query:
                "{original_query}"
                
                {examples}
                
                Provide ONLY the alternative queries, one per line, nothing else:
                """
                
                # Get response from LLM
                response = self.llm(prompt)
                
                # Process response
                if response:
                    # Split by lines and clean up
                    alternatives = [line.strip().strip('"\'') for line in response.strip().split('\n') if line.strip()]
                    
                    # Filter out empty or too short queries
                    alternatives = [q for q in alternatives if len(q) >= 3]
                    
                    # Add to expanded queries
                    expanded_queries.extend(alternatives)
                    
            except Exception as e:
                logger.error(f"Error generating query expansions with LLM: {e}")
        
        return expanded_queries

# Factory function to create query preprocessor
def create_query_preprocessor(llm=None, embedding_model=None) -> QueryPreprocessor:
    """
    Create a query preprocessor with the given models.
    
    Args:
        llm: Language model for query rewriting
        embedding_model: Embedding model for semantic operations
        
    Returns:
        QueryPreprocessor instance
    """
    return QueryPreprocessor(llm=llm, embedding_model=embedding_model) 