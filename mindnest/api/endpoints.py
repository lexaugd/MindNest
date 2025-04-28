"""
API endpoints for MindNest application
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import time

# Import core components
from mindnest.core.document_processor import get_vectorstore
from mindnest.core.llm_manager import get_model_capabilities, llm, qa_chain
from mindnest.utils.query_optimization import process_query
from mindnest.utils.query_cache import query_cache
from mindnest.utils.responses import get_conversation_response
from mindnest.utils.document_compression import optimize_context_for_model
# Import response formatter for formatting and quality control
from mindnest.utils.response_formatter import format_response, validate_response_quality, create_hybrid_response
# Import the new conversational response generator
from mindnest.utils.conversational_response import create_conversational_response_generator, ConversationalResponseGenerator

# Define query types as an Enum for better type safety
class QueryType(Enum):
    CONVERSATION = "CONVERSATION"
    DOCUMENT_QUERY = "DOCUMENT_QUERY" 
    CONCISE_QUERY = "CONCISE_QUERY"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"

# Define router
router = APIRouter()

# Define models
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str
    sources: List[str]
    
class ModelConfig(BaseModel):
    use_small_model: bool = False
    max_context_tokens: int = 800
    conversation_mode: Optional[str] = "professional"  # Add conversation mode option
    query_classifier_mode: Optional[str] = "embeddings"  # Options: "embeddings", "neural", "hybrid", "regex"
    enable_feedback_learning: Optional[bool] = None  # Enable/disable feedback learning

# Helper function to classify queries
def classify_query(query: str) -> QueryType:
    """
    Classify query into one of the supported query types
    """
    try:
        from mindnest.core.config import query_classifier
        
        # First, check for very short, common conversational queries
        # This is a simple pre-filter to ensure chat-like messages are handled conversationally
        if len(query.strip()) <= 3 or query.strip().lower() in ["hi", "hey", "hello", "k", "ok", "bye"]:
            print(f"Pre-filter detected short conversational query: {query}")
            return QueryType.CONVERSATION
        
        if query_classifier:
            # Use the query classifier if available
            result = query_classifier.classify(query)
            
            # Handle tuple return from query classifier
            if isinstance(result, tuple) and len(result) > 0:
                # The first element is the query type
                query_type = result[0]
            else:
                # Fallback if result is not as expected
                query_type = "DOCUMENT_QUERY"
            
            # Convert string to enum
            if query_type == "CONVERSATION":
                return QueryType.CONVERSATION
            elif query_type == "CONCISE_QUERY":
                return QueryType.CONCISE_QUERY
            elif query_type == "DOCUMENT_SEARCH":
                return QueryType.DOCUMENT_SEARCH
            else:
                return QueryType.DOCUMENT_QUERY
        else:
            # Fallback to simple keyword-based classification
            search_keywords = ["search", "find", "locate", "list", "show me"]
            concise_keywords = ["brief", "summarize", "summarise", "short", "concise", "quick", "in a sentence"]
            conversation_keywords = ["hi", "hello", "hey", "greetings", "howdy", "what's up", "good morning", "good afternoon", "good evening"]
            
            # Check for conversation keywords
            if query.lower() in conversation_keywords or len(query.split()) < 3:
                return QueryType.CONVERSATION
            
            # Check for search keywords
            if any(keyword in query.lower() for keyword in search_keywords):
                return QueryType.DOCUMENT_SEARCH
            
            # Check for concise keywords  
            if any(keyword in query.lower() for keyword in concise_keywords):
                return QueryType.CONCISE_QUERY
            
            # Check for conversation
            if len(query.split()) < 5 or query.endswith("?") or "you" in query.lower():
                return QueryType.CONVERSATION
                
            # Default to document query
            return QueryType.DOCUMENT_QUERY
    except Exception as e:
        print(f"Error classifying query: {e}")
        # Default to conversation query for very short queries
        if len(query.split()) < 3:
            print("Falling back to conversation query for short input")
            return QueryType.CONVERSATION
        # Default to document query as fallback for longer queries
        return QueryType.DOCUMENT_QUERY

# Create a conversational response generator
conversational_generator = None
def get_conversational_generator():
    global conversational_generator
    if conversational_generator is None:
        from mindnest.core.llm_manager import llm as system_llm
        conversational_generator = create_conversational_response_generator(llm=system_llm)
    return conversational_generator

# Configuration endpoint
@router.post("/config/model", status_code=200)
async def configure_model(config: ModelConfig):
    """
    Configure the model settings.
    
    Args:
        config: The model configuration
        
    Returns:
        Dict: Confirmation of config change
    """
    import os
    from mindnest.core.config import (
        update_max_context_tokens, update_conversation_mode, 
        update_query_classifier_mode, max_context_tokens,
        conversation_mode, query_classifier_mode, query_classifier
    )
    
    # Update max context tokens
    update_max_context_tokens(config.max_context_tokens)
    
    # Update conversation mode if provided
    if config.conversation_mode:
        if config.conversation_mode in ["professional", "passive_aggressive", "humorous"]:
            update_conversation_mode(config.conversation_mode)
            print(f"Conversation mode updated to: {config.conversation_mode}")
        else:
            raise HTTPException(status_code=400, detail="Invalid conversation mode. Choose 'professional', 'passive_aggressive', or 'humorous'.")
    
    # Update query classifier mode if provided
    if config.query_classifier_mode:
        if config.query_classifier_mode in ["embeddings", "neural", "hybrid", "regex"]:
            update_query_classifier_mode(config.query_classifier_mode)
        else:
            raise HTTPException(status_code=400, detail="Invalid query classifier mode. Choose 'embeddings', 'neural', 'hybrid', or 'regex'.")
    
    # Update feedback learning if specified
    if config.enable_feedback_learning is not None and query_classifier:
        if hasattr(query_classifier, 'set_feedback_learning'):
            query_classifier.set_feedback_learning(config.enable_feedback_learning)
            print(f"Feedback learning {'enabled' if config.enable_feedback_learning else 'disabled'}")
    
    # Update model size if specified
    if config.use_small_model != (os.environ.get("USE_SMALL_MODEL", "").lower() == "true"):
        os.environ["USE_SMALL_MODEL"] = str(config.use_small_model).lower()
        
        # Return a message indicating restart needed for model change
        return {
            "status": "success", 
            "message": "Configuration updated. Server restart required for model change to take effect.",
            "config": {
                "use_small_model": config.use_small_model,
                "max_context_tokens": max_context_tokens,
                "conversation_mode": conversation_mode,
                "query_classifier_mode": query_classifier_mode,
                "feedback_learning_enabled": query_classifier.enable_feedback_learning if query_classifier and hasattr(query_classifier, 'enable_feedback_learning') else False
            }
        }
        
    # Refresh config from global variables to make sure we're getting the latest values
    from mindnest.core.config import max_context_tokens, conversation_mode, query_classifier_mode
        
    return {
        "status": "success", 
        "message": "Configuration updated.",
        "config": {
            "use_small_model": os.environ.get("USE_SMALL_MODEL", "").lower() == "true",
            "max_context_tokens": max_context_tokens,
            "conversation_mode": conversation_mode,
            "query_classifier_mode": query_classifier_mode,
            "feedback_learning_enabled": query_classifier.enable_feedback_learning if query_classifier and hasattr(query_classifier, 'enable_feedback_learning') else False
        }
    }

# Ask question endpoint
@router.post("/ask")
async def ask(request: Request):
    start_time = time.time()  # Track request time
    
    # Get the question from the request
    body = await request.json()
    query = body.get("text", "")
    print(f"Received question: {query}")
    
    # Import global variables directly to ensure we're accessing the most current state
    from mindnest.core.llm_manager import llm, qa_chain, get_model_capabilities, use_small_model, is_llm_ready
    
    # Get model capabilities based on the current model configuration
    model_capabilities = get_model_capabilities()
    print(f"Using model size: {model_capabilities['model_size']}")
    
    try:
        # Check if LLM is properly initialized
        llm_ready = is_llm_ready()
        print(f"LLM ready status: {llm_ready}")
        
        # Handle very short conversational queries immediately without document lookup
        # This ensures that queries like "hi" or "hello" are always treated conversationally
        if len(query.strip()) <= 3 or query.strip().lower() in ["hi", "hey", "hello", "k", "ok", "bye"]:
            print(f"Direct handling of short conversational query: {query}")
            generator = get_conversational_generator()
            response = generator.generate_response(query=query, model_capabilities=model_capabilities)
            return {"text": response, "sources": []}
        
        # Classify query to determine the best approach
        print(f"About to classify query: {query}")
        
        # Get query classification with confidence score
        from mindnest.core.config import query_classifier
        
        # Initialize feedback collector for query classification
        from mindnest.utils.query_classification.feedback import FeedbackCollector
        feedback_collector = FeedbackCollector()
        
        if query_classifier:
            # Use the query classifier if available
            classification_result = query_classifier.classify(query)
            
            # Handle tuple return from query classifier
            if isinstance(classification_result, tuple) and len(classification_result) > 0:
                # The first element is the query type, second is the original query
                query_type_str = classification_result[0]
                confidence = getattr(query_classifier, '_last_confidence', 0.0)
                
                # Convert string to enum
                if query_type_str == "CONVERSATION":
                    query_type = QueryType.CONVERSATION
                elif query_type_str == "CONCISE_QUERY":
                    query_type = QueryType.CONCISE_QUERY
                elif query_type_str == "DOCUMENT_SEARCH":
                    query_type = QueryType.DOCUMENT_SEARCH
                else:
                    query_type = QueryType.DOCUMENT_QUERY
                    
                # Save the classification for potential feedback
                request.state.classification = {
                    "query": query,
                    "predicted_category": query_type_str,
                    "confidence": confidence
                }
            else:
                # Fallback if result is not as expected
                query_type = QueryType.DOCUMENT_QUERY
                request.state.classification = {
                    "query": query,
                    "predicted_category": "DOCUMENT_QUERY",
                    "confidence": 0.0
                }
        else:
            # Simple rule-based classification as fallback
            if query.lower().startswith("find "):
                query_type = QueryType.DOCUMENT_SEARCH
            elif len(query.split()) < 4:
                query_type = QueryType.CONVERSATION
            elif any(term in query.lower() for term in ["summarize", "brief", "short", "tldr", "concise"]):
                query_type = QueryType.CONCISE_QUERY
            else:
                query_type = QueryType.DOCUMENT_QUERY
                
            request.state.classification = {
                "query": query,
                "predicted_category": query_type.name,
                "confidence": 0.0
            }
                
        print(f"Query classified as: {query_type}")
        
        # Different handling based on query type
        if query_type == QueryType.CONVERSATION:
            # Implicit positive feedback for conversation queries - they're usually obvious
            if request.state.classification.get("confidence", 0) > 0.8:
                feedback_collector.add_feedback(
                    query=query,
                    predicted_category="CONVERSATION",
                    confidence=request.state.classification.get("confidence", 0.9),
                    is_correct=True
                )
            
            print("Processing as conversational query")
            # For conversational queries, we use the new ConversationalResponseGenerator
            if not llm_ready:
                print("LLM not initialized, providing fallback response")
                fallback_response = "I'm unable to process your question at the moment because the language model is still initializing. Please try again in a few moments, or try a different type of query that doesn't require the language model."
                return {"text": fallback_response, "sources": []}
            
            try:
                # Get conversation mode from config
                from mindnest.core.config import conversation_mode
                
                print(f"Using conversation mode: {conversation_mode}")
                
                # Get the conversational response generator
                generator = get_conversational_generator()
                
                # Use the generator to create a response
                response = generator.generate_response(
                    query=query, 
                    model_capabilities=model_capabilities
                )
                
                # Apply conversation mode styling if needed
                if conversation_mode != "professional":
                    # Get the styled prefix
                    styled_prefix = get_conversation_response(
                        mode=conversation_mode, 
                        query=query
                    ).split(".")[0] + ". "
                    
                    # Combine the styled prefix with the generated response if needed
                    if not any(greeting in response for greeting in ["Hello!", "Hi!", "Hey!"]):
                        response = styled_prefix + response
                
                return {"text": response, "sources": []}
            except Exception as e:
                print(f"Error in conversation processing: {e}")
                import traceback
                traceback.print_exc()
                fallback_response = "An error occurred while processing your conversational query."
                return {"text": fallback_response, "sources": []}
                
        # For document-based queries, first check if we have documents
        vectorstore = get_vectorstore()
        
        # Get document count to verify we have documents
        try:
            doc_count = len(vectorstore.get()["ids"])
            print(f"Vector store contains {doc_count} document chunks")
            
            # If no documents, default to conversation mode
            if doc_count == 0:
                print("No documents found in vector store, falling back to conversational mode")
                generator = get_conversational_generator()
                response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                return {"text": response, "sources": []}
        except Exception as e:
            print(f"Error checking vector store document count: {e}")
            # If we can't check, proceed with normal processing
        
        if query_type == QueryType.CONCISE_QUERY:
            print("Processing as concise RAG query")
            # For concise queries, we retrieve fewer documents
            # Use model-specific document limit for concise queries
            k = model_capabilities["concise_limit"]
            print(f"Retrieving {k} documents based on model capabilities")
            
            try:
                # Check if we have cached document retrieval results
                if query in query_cache:
                    print("Using cached document retrieval results")
                    docs = query_cache.get(query)
                else:
                    vectorstore = get_vectorstore()
                    docs = vectorstore.similarity_search(query, k=k)
                    query_cache.set(query, docs)
                
                print(f"Found {len(docs)} relevant documents")
                
                # If no relevant documents found, fall back to conversational mode
                if not docs:
                    print("No relevant documents found, falling back to conversational mode")
                    generator = get_conversational_generator()
                    response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                    return {"text": response, "sources": []}
                
                # Optimize context for model size
                docs = optimize_context_for_model(docs, query, model_capabilities)
                
                # If LLM is not ready, return the documents directly
                if not llm_ready:
                    print("LLM not initialized, providing document retrieval only")
                    fallback_response = "I'm unable to process your question at the moment because the language model is still initializing. Here are the most relevant document snippets I found."
                    return {"text": fallback_response, "sources": [doc.metadata["source"] for doc in docs[:3] if hasattr(doc, 'metadata')]}
                
                # Create custom chain with model-specific prompt
                print("Creating QA chain with concise prompt")
                try:
                    from langchain.chains.question_answering import load_qa_chain
                    from mindnest.core.llm_manager import get_model_specific_prompts
                    model_prompts = get_model_specific_prompts(model_capabilities)
                    custom_chain = load_qa_chain(llm, chain_type="stuff", prompt=model_prompts["concise"])
                    
                    print("Invoking QA chain")
                    answer = custom_chain.invoke({
                        "input_documents": docs,
                        "query": query
                    })
                    
                    # Format response for concise queries
                    print("Formatting response")
                    formatted_response = format_response(answer["output_text"], model_capabilities)
                    validated_response = validate_response_quality(formatted_response, query, model_capabilities)
                    
                    end_time = time.time()
                    print(f"Request processed in {end_time - start_time:.2f} seconds")
                    return {"text": validated_response, "sources": [doc.metadata["source"] for doc in docs]}
                except Exception as e:
                    print(f"Error in concise query processing: {e}")
                    import traceback
                    traceback.print_exc()
                    fallback_response = "An error occurred while processing your concise query. Here are the most relevant documents I found."
                    return {"text": fallback_response, "sources": [doc.metadata["source"] for doc in docs[:3] if hasattr(doc, 'metadata')]}
            except Exception as e:
                print(f"Error in document retrieval for concise query: {e}")
                import traceback
                traceback.print_exc()
                
                # Fall back to conversational mode if document retrieval fails
                print("Document retrieval failed, falling back to conversational mode")
                generator = get_conversational_generator()
                response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                return {"text": response, "sources": []}
        
        elif query_type == QueryType.DOCUMENT_QUERY:
            print("Processing as standard RAG query")
            try:
                # Check if we have cached document retrieval results
                if query in query_cache:
                    print("Using cached document retrieval results")
                    docs = query_cache.get(query)
                else:
                    # Use model-specific document limit for regular queries
                    k = model_capabilities["document_limit"]
                    print(f"Retrieving {k} documents based on model capabilities")
                    vectorstore = get_vectorstore()
                    docs = vectorstore.similarity_search(query, k=k)
                    query_cache.set(query, docs)
                
                print(f"Found {len(docs)} relevant documents")
                
                # If no relevant documents found, fall back to conversational mode
                if not docs:
                    print("No relevant documents found, falling back to conversational mode")
                    generator = get_conversational_generator()
                    response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                    return {"text": response, "sources": []}
                
                # Optimize context for model size
                print("Optimizing context for model size")
                docs = optimize_context_for_model(docs, query, model_capabilities)
                
                # If LLM is not ready, return the documents directly
                if not llm_ready:
                    print("LLM not initialized, providing document retrieval only")
                    fallback_response = "I'm unable to process your question at the moment because the language model is still initializing. Here are the most relevant document snippets I found."
                    return {"text": fallback_response, "sources": [doc.metadata["source"] for doc in docs[:3] if hasattr(doc, 'metadata')]}
                
                try:
                    # Create a new chain specifically for this request
                    print("Creating QA chain with document prompt")
                    from langchain.chains.question_answering import load_qa_chain
                    from langchain.prompts import PromptTemplate
                    from mindnest.core.llm_manager import get_model_specific_prompts
                    
                    model_prompts = get_model_specific_prompts(model_capabilities)
                    custom_chain = load_qa_chain(llm, chain_type="stuff", prompt=model_prompts["document"])
                    
                    print("Invoking QA chain")
                    answer = custom_chain.invoke({
                        "input_documents": docs,
                        "query": query
                    })
                    
                    # Apply model-specific formatting and quality control
                    print("Formatting and validating response")
                    formatted_response = format_response(answer["output_text"], model_capabilities)
                    validated_response = validate_response_quality(formatted_response, query, model_capabilities)
                    final_response = create_hybrid_response(validated_response, query, docs, model_capabilities)
                    
                    end_time = time.time()
                    print(f"Request processed in {end_time - start_time:.2f} seconds")
                    return {"text": final_response, "sources": [doc.metadata["source"] for doc in docs]}
                except Exception as e:
                    print(f"Error during QA chain invocation: {e}")
                    import traceback
                    traceback.print_exc()
                    fallback_response = "An error occurred while processing your query. Here are the most relevant documents I found."
                    return {"text": fallback_response, "sources": [doc.metadata["source"] for doc in docs[:3] if hasattr(doc, 'metadata')]}
            except Exception as e:
                print(f"Error in document retrieval for document query: {e}")
                import traceback
                traceback.print_exc()
                
                # Fall back to conversational mode if document retrieval fails
                print("Document retrieval failed, falling back to conversational mode")
                generator = get_conversational_generator()
                response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                return {"text": response, "sources": []}
        else:
            # Handle document search type queries
            print("Processing as document search query")
            try:
                k = model_capabilities["document_limit"]
                print(f"Retrieving {k} documents based on model capabilities")
                vectorstore = get_vectorstore()
                docs = vectorstore.similarity_search(query, k=k)
                
                # If no relevant documents found, fall back to conversational mode
                if not docs:
                    print("No relevant documents found, falling back to conversational mode")
                    generator = get_conversational_generator()
                    response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                    return {"text": response, "sources": []}
                
                content = f"Found {len(docs)} relevant documents for: '{query}'\n\nSnippets:\n"
                for doc in docs:
                    content += f"- {doc.metadata['source']}: {doc.page_content[:200]}...\n"
                
                end_time = time.time()
                print(f"Request processed in {end_time - start_time:.2f} seconds")
                return {"text": content, "sources": [doc.metadata["source"] for doc in docs]}
            except Exception as e:
                print(f"Error in document search processing: {e}")
                import traceback
                traceback.print_exc()
                
                # Fall back to conversational mode if document search fails
                print("Document search failed, falling back to conversational mode")
                generator = get_conversational_generator()
                response = generator.generate_response(query=query, model_capabilities=model_capabilities)
                return {"text": response, "sources": []}
    
    except Exception as e:
        print(f"Top-level error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        # Always fall back to conversational mode for any error
        try:
            print("Major error, falling back to basic conversational mode")
            generator = get_conversational_generator()
            response = generator.generate_response(query=query, model_capabilities=model_capabilities)
            return {"text": response, "sources": []}
        except:
            # Last resort fallback
            fallback_response = "I'm currently experiencing technical difficulties. Please try again with a simpler query."
            return {"text": fallback_response, "sources": []}

# Simple test endpoint for debugging
@router.post("/test-ask")
async def test_ask(request: Request):
    """A simplified version of the ask endpoint for debugging."""
    try:
        # Get the question from the request
        body = await request.json()
        query = body.get("text", "")
        print(f"Received test question: {query}")
        
        # Import global variables directly to ensure we're accessing the most current state
        from mindnest.core.llm_manager import llm, get_model_capabilities, is_llm_ready
        
        # Get model capabilities
        model_capabilities = get_model_capabilities()
        print(f"Using model size: {model_capabilities['model_size']}")
        
        # Get vector store and retrieve documents
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)
        
        # Check if llm is ready
        if is_llm_ready():
            print("LLM is initialized and ready")
            # Create simple prompt with context and query
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            
            # Get response from LLM
            try:
                response = llm.invoke(prompt)
                return {"text": response, "sources": [doc.metadata["source"] for doc in docs]}
            except Exception as e:
                print(f"Error invoking LLM: {e}")
                import traceback
                traceback.print_exc()
                return {"text": f"Error processing with LLM: {str(e)}", "sources": [doc.metadata["source"] for doc in docs]}
        else:
            print("LLM not initialized")
            return {
                "text": "The LLM is not initialized. Here are the most relevant documents I found:",
                "sources": [doc.metadata["source"] for doc in docs],
                "raw_content": [doc.page_content for doc in docs]
            }
            
    except Exception as e:
        print(f"Error in test_ask: {e}")
        import traceback
        traceback.print_exc()
        return {"text": f"Error: {str(e)}", "sources": []}

# Feedback endpoint for query classification
@router.post("/feedback/classification")
async def classification_feedback(request: Request):
    """
    Endpoint for collecting feedback on query classification.
    
    This allows users to provide feedback when they believe a query was
    misclassified, which helps improve the classifier over time.
    """
    try:
        body = await request.json()
        
        # Extract feedback data
        query = body.get("query", "")
        predicted_category = body.get("predicted_category", "")
        correct_category = body.get("correct_category", "")
        confidence = float(body.get("confidence", 0.0))
        
        # Validate input
        if not query or not predicted_category or not correct_category:
            return {"status": "error", "message": "Missing required fields"}
            
        # Initialize feedback collector
        from mindnest.utils.query_classification.feedback import FeedbackCollector
        feedback_collector = FeedbackCollector()
        
        # Determine if the prediction was correct
        is_correct = predicted_category == correct_category
        
        # Add feedback
        feedback_collector.add_feedback(
            query=query,
            predicted_category=predicted_category,
            confidence=confidence,
            correct_category=correct_category,
            is_correct=is_correct
        )
        
        # Update classifier if applicable
        from mindnest.core.config import query_classifier
        classifier_updated = False
        if query_classifier and hasattr(query_classifier, 'update_from_feedback'):
            try:
                # Only force update if the classification was incorrect
                # Otherwise, let the regular interval-based updates handle it
                force_update = not is_correct
                classifier_updated = query_classifier.update_from_feedback(force=force_update)
            except Exception as e:
                logger.error(f"Error updating classifier from feedback: {e}")
        
        # Get current statistics
        stats = feedback_collector.get_stats()
        
        response = {
            "status": "success", 
            "message": "Feedback recorded successfully",
            "stats": {
                "total_feedback": stats["total_feedback"],
                "accuracy": stats.get("accuracy", 0.0),
                "correct_classifications": stats["correct_classifications"],
                "incorrect_classifications": stats["incorrect_classifications"]
            }
        }
        
        if classifier_updated:
            response["classifier_updated"] = True
            
        return response
        
    except Exception as e:
        print(f"Error processing classification feedback: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"}

# Feedback statistics endpoint
@router.get("/feedback/stats")
async def feedback_stats():
    """
    Get statistics about classification feedback.
    """
    try:
        # Initialize feedback collector
        from mindnest.utils.query_classification.feedback import FeedbackCollector
        feedback_collector = FeedbackCollector()
        
        # Get current statistics
        stats = feedback_collector.get_stats()
        
        # Add query classifier stats if available
        from mindnest.core.config import query_classifier
        if query_classifier:
            classifier_stats = query_classifier.get_stats()
            combined_stats = {
                "feedback": stats,
                "classifier": classifier_stats
            }
            return combined_stats
        
        return {"feedback": stats}
    except Exception as e:
        print(f"Error getting feedback statistics: {e}")
        return {"error": str(e), "status": "Could not retrieve complete statistics"}

# Document listing endpoint
@router.get("/documents")
async def list_documents():
    """List all documents in the vector store."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Get all documents from the vectorstore
        all_docs = vectorstore.get()
        total_count = len(all_docs["ids"])
        
        # Extract unique sources for reporting
        sources = []
        if "metadatas" in all_docs and all_docs["metadatas"]:
            for metadata in all_docs["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    if source not in sources:
                        sources.append(source)
        
        return {
            "total_documents": total_count,
            "sources": sources
        }
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System statistics endpoint
@router.get("/stats")
async def get_stats():
    """Get statistics about the document processing system."""
    try:
        # Create a document tracker to get document statistics
        from mindnest.utils.document_tracker import DocumentTracker
        tracker = DocumentTracker()
        
        # Get cache statistics
        cache_stats = query_cache.get_stats()
        
        # Get document statistics
        doc_count = tracker.get_document_count()
        extensions = tracker.get_file_extensions()
        last_update = tracker.get_last_update()
        
        # Get vector store statistics
        vs_stats = {}
        vectorstore = get_vectorstore()
        if vectorstore:
            vs_data = vectorstore.get()
            vs_stats = {
                "total_chunks": len(vs_data["ids"]),
                "embedding_dim": len(vs_data["embeddings"][0]) if vs_data["embeddings"] else 0,
            }
            
        # Get query classifier statistics
        classifier_stats = {}
        from mindnest.core.config import query_classifier
        if query_classifier is not None:
            classifier_stats = query_classifier.get_stats()
        
        return {
            "documents": {
                "count": doc_count,
                "extensions": extensions,
                "last_update": last_update
            },
            "cache": {
                "memory_hits": cache_stats["memory_hits"],
                "disk_hits": cache_stats["disk_hits"],
                "misses": cache_stats["misses"],
                "total_queries": cache_stats["total_queries"],
                "hit_rate_percent": cache_stats["hit_rate_percent"],
                "memory_size": cache_stats["memory_cache_size"],
                "disk_size": cache_stats["disk_cache_size"]
            },
            "vectorstore": vs_stats,
            "query_classifier": classifier_stats
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {
            "error": str(e),
            "status": "Could not retrieve complete statistics"
        }

# Cache clearing endpoints
@router.post("/clear-cache")
async def clear_cache():
    """Clear the query cache."""
    try:
        query_cache.clear()
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@router.post("/clear-classifier-cache")
async def clear_classifier_cache():
    """Clear the query classifier cache."""
    from mindnest.core.config import query_classifier
    
    try:
        if query_classifier is not None:
            query_classifier.clear_cache()
            return {"status": "success", "message": "Query classifier cache cleared successfully"}
        else:
            return {"status": "warning", "message": "No query classifier initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing classifier cache: {str(e)}")

# Get system configuration
@router.get("/config")
async def get_config():
    """
    Get current system configuration.
    
    Returns:
        Dict: Current configuration values
    """
    import os
    from mindnest.core.config import max_context_tokens, conversation_mode, query_classifier_mode
    
    document_info = {}  # This will need to be updated to get from document tracker
    
    return {
        "status": "success",
        "config": {
            "use_small_model": os.environ.get("USE_SMALL_MODEL", "").lower() == "true",
            "max_context_tokens": max_context_tokens,
            "conversation_mode": conversation_mode,
            "query_classifier_mode": query_classifier_mode,
            "model_name": os.environ.get("MODEL_NAME", "unknown"),
            "document_count": len(document_info) if document_info else 0
        }
    }

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running."""
    from mindnest.core.llm_manager import llm, model_name, small_model_name, use_small_model
    
    llm_status = "loaded" if llm is not None else "not loaded"
    vectorstore = get_vectorstore()
    vectorstore_status = "loaded" if vectorstore is not None else "not loaded"
    current_model = small_model_name if use_small_model else model_name
    return {
        "status": "healthy", 
        "llm": llm_status,
        "vectorstore": vectorstore_status,
        "model": current_model
    }

# Training endpoint for query classifier
@router.post("/classifier/train")
async def train_classifier(request: Request):
    """
    Endpoint for training the query classifier with feedback data.
    
    This allows the system to periodically train the classifier based on
    feedback collected from users, improving its accuracy over time.
    """
    try:
        # Get request parameters
        body = await request.json()
        min_confidence = float(body.get("min_confidence", 0.8))
        min_samples = int(body.get("min_samples", 10))
        
        # Make sure we have a classifier to train
        from mindnest.core.config import query_classifier
        if not query_classifier:
            return {"status": "error", "message": "No query classifier available to train"}
            
        # Check if we're using a classifier that can be trained with feedback
        if query_classifier.classifier_type not in ["zero-shot", "embedding"]:
            return {
                "status": "error", 
                "message": f"Classifier type '{query_classifier.classifier_type}' cannot be trained from feedback"
            }
            
        # Initialize feedback collector
        from mindnest.utils.query_classification.feedback import FeedbackCollector
        feedback_collector = FeedbackCollector()
        
        # Get high confidence correct classifications
        all_entries = feedback_collector.get_feedback_entries(limit=1000)
        
        # Filter for correct classifications with high confidence
        valid_entries = [
            entry for entry in all_entries
            if entry.get("is_correct") and entry.get("confidence", 0) >= min_confidence
        ]
        
        if len(valid_entries) < min_samples:
            return {
                "status": "warning",
                "message": f"Not enough valid samples for training. Found {len(valid_entries)}, need at least {min_samples}."
            }
            
        # Group by category
        entries_by_category = {}
        for entry in valid_entries:
            category = entry["predicted_category"]
            if category not in entries_by_category:
                entries_by_category[category] = []
            entries_by_category[category].append(entry)
            
        # If we're using a ZeroShotClassifier, we can add new descriptions
        if query_classifier.classifier_type == "zero-shot" and hasattr(query_classifier.classifier, "add_descriptions"):
            # Add new descriptions based on feedback
            for category, entries in entries_by_category.items():
                if len(entries) >= 5:  # Require at least 5 samples per category
                    query_classifier.classifier.add_descriptions(category, [e["query"] for e in entries[:10]])
                    
            # Reinitialize after adding descriptions
            query_classifier.classifier.initialized = False
            query_classifier.classifier.initialize()
            
            return {
                "status": "success",
                "message": "Added new descriptions to zero-shot classifier",
                "stats": {
                    "total_entries": len(valid_entries),
                    "categories_updated": len(entries_by_category),
                    "samples_by_category": {k: len(v) for k, v in entries_by_category.items()}
                }
            }
            
        # For other classifiers, we can provide the feedback data in a generic way
        total_trained = 0
        if hasattr(query_classifier.classifier, "train_from_feedback"):
            total_trained = query_classifier.classifier.train_from_feedback(valid_entries)
            
            return {
                "status": "success",
                "message": f"Trained classifier with {total_trained} examples",
                "stats": {
                    "total_entries": len(valid_entries),
                    "total_trained": total_trained,
                    "categories": list(entries_by_category.keys())
                }
            }
            
        return {
            "status": "error",
            "message": "Current classifier doesn't support training from feedback"
        }
        
    except Exception as e:
        print(f"Error training classifier: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"}

@router.post("/classifier/update-from-feedback")
async def update_classifier_from_feedback():
    """
    Manually update the query classifier based on collected feedback.
    
    This endpoint forces the classifier to learn from the feedback data,
    updating its examples and potentially improving future classifications.
    """
    try:
        from mindnest.core.config import query_classifier
        
        if not query_classifier:
            return {
                "status": "error", 
                "message": "Query classifier is not initialized"
            }
            
        if not hasattr(query_classifier, 'update_from_feedback'):
            return {
                "status": "error", 
                "message": "Query classifier does not support feedback learning"
            }
            
        # Force update from feedback
        updated = query_classifier.update_from_feedback(force=True)
        
        if updated:
            # Get classifier stats after update
            stats = query_classifier.get_stats()
            
            # If feedback learning stats are available, include them
            feedback_learning_stats = {}
            if "feedback_learning" in stats:
                feedback_learning_stats = stats["feedback_learning"]
                
            return {
                "status": "success",
                "message": "Classifier updated successfully from feedback data",
                "updated": True,
                "feedback_learning_stats": feedback_learning_stats
            }
        else:
            return {
                "status": "warning",
                "message": "No updates were made to the classifier",
                "updated": False,
                "reason": "No suitable feedback entries found or feedback learning is disabled"
            }
            
    except Exception as e:
        print(f"Error updating classifier from feedback: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"}

@router.get("/classifier/examples")
async def get_classifier_examples():
    """
    Get the example queries that the classifier is using.
    
    This endpoint returns all the examples used by the classifier,
    including those learned from user feedback.
    """
    try:
        from mindnest.core.config import query_classifier
        
        if not query_classifier:
            return {
                "status": "error", 
                "message": "Query classifier is not initialized"
            }
            
        # Check if classifier has examples available
        if not hasattr(query_classifier, 'classifier') or not hasattr(query_classifier.classifier, 'examples'):
            return {
                "status": "error", 
                "message": "Classifier does not have accessible examples"
            }
            
        # Get the examples
        examples = query_classifier.classifier.examples
        
        # Get feedback learning status
        feedback_learning_enabled = (
            hasattr(query_classifier, 'enable_feedback_learning') and 
            query_classifier.enable_feedback_learning
        )
        
        # Get feedback learning stats if available
        feedback_learning_stats = {}
        if hasattr(query_classifier, 'feedback_learner') and query_classifier.feedback_learner:
            feedback_learning_stats = query_classifier.feedback_learner.get_stats()
            
        # Count examples per category
        category_counts = {
            category: len(queries) 
            for category, queries in examples.items()
        }
            
        return {
            "status": "success",
            "examples": examples,
            "category_counts": category_counts,
            "feedback_learning_enabled": feedback_learning_enabled,
            "feedback_learning_stats": feedback_learning_stats
        }
            
    except Exception as e:
        print(f"Error getting classifier examples: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"} 