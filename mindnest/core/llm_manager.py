"""
LLM initialization and management for MindNest application
"""

import os
from typing import Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize global variables
model_name = "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf"  # Default model
small_model_name = "llama-2-7b.Q4_K_M.gguf"  # Smaller, faster model option
use_small_model = os.environ.get("USE_SMALL_MODEL", "false").lower() == "true"
llm = None
qa_chain = None
embeddings = None

# Added to track initialization status
is_llm_initialized = False
is_qa_chain_initialized = False

def get_llm():
    """Get the current LLM instance."""
    global llm
    return llm

def get_qa_chain():
    """Get the current QA chain instance."""
    global qa_chain
    return qa_chain

def is_llm_ready():
    """Check if the LLM is initialized and ready for use."""
    global llm, is_llm_initialized
    return llm is not None and is_llm_initialized

def initialize_embeddings():
    """Initialize embeddings for vector search and query classification."""
    global embeddings
    
    try:
        print("Initializing embeddings...")
        # Try to import from the updated package first
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            # Fallback to community package if langchain_huggingface is not available
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        try:
            # Fallback to basic embeddings
            from langchain_community.embeddings import FakeEmbeddings
            print("Falling back to simple embeddings")
            embeddings = FakeEmbeddings(size=384)
            return embeddings
        except Exception as e2:
            print(f"Fatal error initializing embeddings: {e2}")
            raise

def initialize_llm():
    """Initialize the LLM model separately."""
    global llm, model_name, use_small_model, small_model_name, is_llm_initialized
    
    try:
        from langchain_community.llms import LlamaCpp
        
        print("Initializing LLM...")
        
        # Choose model based on configuration
        selected_model = small_model_name if use_small_model else model_name
        model_path = os.path.abspath(f"models/{selected_model}")
        
        print(f"Loading model: {selected_model} from: {model_path}")
        
        # Set n_ctx and batch size based on model size
        n_ctx = 2048 if use_small_model else 4096  # Smaller context for smaller model
        n_batch = 1024 if use_small_model else 1024  # Can be faster for smaller models
        
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.3,
            max_tokens=2000,
            top_p=0.95,
            verbose=True,
            n_ctx=n_ctx,
            n_batch=n_batch,  # Increased from 512 to 1024 for faster processing
            n_gpu_layers=40,  # Use GPU acceleration if available
            repeat_penalty=1.1,
            f16_kv=True,
            use_mlock=True,  # Keep model in memory
            seed=42,  # Consistent results
            logits_all=False,  # Don't compute logits for all tokens (speeds up)
            stop=["</s>"],  # Stop token for faster completion
        )
        print(f"LLM initialized successfully with {n_ctx} context window and {n_batch} batch size")
        is_llm_initialized = True
        return True
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        is_llm_initialized = False
        return False

def initialize_qa_chain():
    """Initialize the question-answering chain with the LLM."""
    global qa_chain, llm, is_qa_chain_initialized
    
    try:
        if llm is None:
            print("LLM not initialized. QA chain initialization skipped.")
            is_qa_chain_initialized = False
            return False
    
        print("Initializing QA chain...")
        
        from langchain.chains.question_answering import load_qa_chain
        from langchain.prompts import PromptTemplate
        
        # Model-specific prompt templates
        if use_small_model:
            # Structured, simpler template for small models
            template = """
            Answer the question based ONLY on the context provided below.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {query}
            
            INSTRUCTIONS:
            - Use ONLY information from the context
            - Keep your answer clear and direct
            - If the question asks for a brief answer, be very concise
            - Format your answer in simple paragraphs
            - If you don't know, say "The documentation doesn't provide this information"
            
            ANSWER:
            """
        else:
            # More flexible template for larger models
            template = """
            Answer the following question based on the provided context. 
            
            If the question asks for a brief or concise answer, keep your response short and to the point.
            If the question asks for a summary or a definition in one sentence, provide exactly that.
            Focus on answering the exact question without adding irrelevant information.
            Only include information that is directly relevant to answering the specific question.
            If the information isn't in the context, acknowledge that the documentation doesn't cover it.
            
            Context pieces:
            {context}
            
            Question: {query}
            
            Answer:
            """
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Create the chain
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
        
        print("QA chain initialized successfully with model-specific template")
        is_qa_chain_initialized = True
        return True
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        import traceback
        traceback.print_exc()
        is_qa_chain_initialized = False
        return False

def get_model_capabilities() -> Dict[str, Any]:
    """
    Get a dictionary of model capabilities for context-aware processing.
    
    Returns:
        Dict[str, Any]: Dictionary with model capabilities
    """
    # Determine model size based on configuration
    model_size = "small" if use_small_model else "large"
    
    # Set context window size and other capabilities based on model size
    if model_size == "small":
        # Small/lightweight model (e.g., llama-2-7b)
        return {
            "model_size": "small",
            "context_window": 2048,  # Typical context window for 7B models
            "document_limit": 3,     # Retrieve fewer documents for small models
            "concise_limit": 2,      # Be more selective for concise queries
            "max_response_length": 250,  # Keep responses shorter for small models
            "prefers_structured_prompts": True,  # Small models do better with structure
            "context_compression_pct": 0.70,  # More aggressive compression
            "optimal_token_limit": 1536  # Lower token limit for small models
        }
    else:
        # Large model (e.g., Wizard-Vicuna-13B)
        return {
            "model_size": "large", 
            "context_window": 8192,  # Higher context window for 13B+ models
            "document_limit": 5,     # Can handle more documents
            "concise_limit": 3,      # Can handle more documents for concise queries
            "max_response_length": 500,  # Can generate longer responses
            "prefers_structured_prompts": False,  # More flexible with prompt format
            "context_compression_pct": 0.85,  # Less aggressive compression
            "optimal_token_limit": 6144  # Higher token limit for larger models
        }

def get_model_specific_prompts(model_capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get prompt templates optimized for the current model size.
    
    Args:
        model_capabilities (dict): Model capabilities dictionary
        
    Returns:
        dict: Dictionary of prompt templates for different query types
    """
    from langchain.prompts import PromptTemplate
    
    model_size = model_capabilities["model_size"]
    
    if model_size == "small":
        # Simpler, more structured prompts for small models
        concise_template = """
        Answer the question below using ONLY the provided context information.
        Keep your answer in 1-2 short sentences.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        SHORT ANSWER:
        """
        
        document_template = """
        Answer the following question using ONLY information from the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        ANSWER (be clear and direct):
        """
    else:
        # More flexible prompts for larger models
        concise_template = """
        Answer the following question in a single concise paragraph of at most 2-3 sentences.
        Be direct, clear, and brief without unnecessary details.
        Use only the information provided in the context.
        
        Context information:
        {context}
        
        Question: {query}
        
        Concise Answer (2-3 sentences maximum):
        """
        
        document_template = """
        Answer the following question based on the provided context. 
        
        If the question asks for a brief or concise answer, keep your response short and to the point.
        If the question asks for a summary or a definition in one sentence, provide exactly that.
        Focus on answering the exact question without adding irrelevant information.
        Only include information that is directly relevant to answering the specific question.
        
        Context pieces:
        {context}
        
        Question: {query}
        
        Answer:
        """
    
    return {
        "concise": PromptTemplate.from_template(concise_template),
        "document": PromptTemplate.from_template(document_template)
    }

def cleanup_resources():
    """Clean up resources related to LLM and embeddings."""
    global llm, qa_chain, embeddings
    
    print("Shutting down and cleaning up resources...")
    
    # Clean up query classifier
    from mindnest.core.config import query_classifier
    if query_classifier is not None:
        try:
            query_classifier.clear_cache()
            print("Query classifier cache cleared")
        except Exception as e:
            print(f"Warning: Error cleaning up query classifier: {e}")
    
    # Clean up LLM
    if llm is not None:
        try:
            # Some LLMs have cleanup methods
            if hasattr(llm, "cleanup"):
                llm.cleanup()
        except Exception as e:
            print(f"Warning: Error cleaning up LLM: {e}")
    
    # Clean up any semaphores
    try:
        import multiprocessing
        if hasattr(multiprocessing, "_resource_tracker") and hasattr(multiprocessing._resource_tracker, "_resource_tracker"):
            # Force the resource tracker to clean up
            multiprocessing.resource_tracker._resource_tracker._check_alive()
    except Exception as e:
        print(f"Warning: Error cleaning up multiprocessing resources: {e}")
    
    # Set to None to help garbage collection
    llm = None
    qa_chain = None 