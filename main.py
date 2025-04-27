import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PythonLoader
)
from fastapi.requests import Request
import shutil
import re
import random
import time

# Import our optimized components
from utils.responses import get_conversation_response, assess_query_complexity
from utils.query_optimization import categorize_query
from utils.query_cache import QueryCache
from utils.incremental_vectorstore import IncrementalVectorStore
from utils.llm_manager import llm_manager

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Catch-all route for 404 errors
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"The requested URL {request.url.path} was not found on this server. Please check your URL and try again."}
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running."""
    llm_status = "loaded" if llm is not None else "not loaded"
    vectorstore_status = "loaded" if vectorstore is not None else "not loaded"
    current_model = small_model_name if use_small_model else model_name
    return {
        "status": "healthy", 
        "llm": llm_status,
        "vectorstore": vectorstore_status,
        "model": current_model
    }

# Initialize embeddings with error handling and fallback options
try:
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings initialized successfully")
except Exception as e:
    print(f"Error initializing embeddings: {e}")
    try:
        # Fallback to basic embeddings
        from langchain_community.embeddings import FakeEmbeddings
        print("Falling back to simple embeddings")
        embeddings = FakeEmbeddings(size=384)
    except Exception as e2:
        print(f"Fatal error initializing embeddings: {e2}")
        raise

# Initialize variables
vectorstore = None
llm = None
qa_chain = None
model_name = "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf"  # Default model
small_model_name = "llama-2-7b.Q4_K_M.gguf"  # Smaller, faster model option
use_small_model = os.environ.get("USE_SMALL_MODEL", "false").lower() == "true"
max_context_tokens = 800  # Cap context for faster responses
query_cache = QueryCache()  # Initialize query cache
conversation_mode = "professional"  # Default to professional mode
document_info = {}
query_classifier_mode = "embeddings"  # Default query classifier mode
query_classifier = None

# Add the get_model_specific_prompts function
def get_model_specific_prompts(model_capabilities):
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

def initialize_vectorstore():
    """Initialize the vector store with documents."""
    global vectorstore
    try:
        print("Initializing vector store using incremental implementation...")
        # Use our optimized incremental vector store implementation
        inc_vectorstore = IncrementalVectorStore()
        
        # Option to force rebuild the vector store
        force_rebuild = False
        
        # Initialize or update the vector store
        vectorstore = inc_vectorstore.initialize_or_update(force_rebuild=force_rebuild)
        
        if vectorstore is not None:
            print(f"Vector store ready with {len(vectorstore.get()['ids'])} documents")
        else:
            print("Failed to initialize vector store")
            
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

def initialize_llm():
    """Initialize the LLM model separately."""
    global llm, model_name, use_small_model, small_model_name
    
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
        return True
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return False

def initialize_qa_chain():
    """Initialize the question-answering chain with the LLM."""
    global qa_chain, llm, max_context_tokens, use_small_model
    
    try:
        if llm is None:
            print("LLM not initialized. QA chain initialization skipped.")
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
        return True
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    
@app.post("/config/model", status_code=200)
async def configure_model(config: ModelConfig):
    """
    Configure the model settings.
    
    Args:
        config: The model configuration
        
    Returns:
        Dict: Confirmation of config change
    """
    global max_context_tokens, conversation_mode, query_classifier_mode
    max_context_tokens = config.max_context_tokens
    
    # Update conversation mode if provided
    if config.conversation_mode:
        if config.conversation_mode in ["professional", "passive_aggressive", "humorous"]:
            conversation_mode = config.conversation_mode
        else:
            raise HTTPException(status_code=400, detail="Invalid conversation mode. Choose 'professional', 'passive_aggressive', or 'humorous'.")
    
    # Update query classifier mode if provided
    if config.query_classifier_mode:
        if config.query_classifier_mode in ["embeddings", "neural", "hybrid", "regex"]:
            query_classifier_mode = config.query_classifier_mode
            # Reinitialize the query classifier if it exists
            if 'query_classifier' in globals() and query_classifier is not None:
                query_classifier.classifier_type = query_classifier_mode
                query_classifier.classifier = None  # Force reinitialization
                query_classifier.initialize()
        else:
            raise HTTPException(status_code=400, detail="Invalid query classifier mode. Choose 'embeddings', 'neural', 'hybrid', or 'regex'.")
    
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
                "query_classifier_mode": query_classifier_mode
            }
        }
        
    return {
        "status": "success", 
        "message": "Configuration updated.",
        "config": {
            "use_small_model": os.environ.get("USE_SMALL_MODEL", "").lower() == "true",
            "max_context_tokens": max_context_tokens,
            "conversation_mode": conversation_mode,
            "query_classifier_mode": query_classifier_mode
        }
    }

def process_query(text, model_capabilities=None):
    """
    Process the query and determine if it should be handled as a search or normal query.
    
    Args:
        text (str): The user's query text
        model_capabilities (dict, optional): Model capabilities for model-aware classification
        
    Returns:
        tuple: Query type and processed query
    """
    global query_classifier, query_classifier_mode
    
    # Get model size for optimized classification
    model_size = "large"  # Default
    if model_capabilities and "model_size" in model_capabilities:
        model_size = model_capabilities["model_size"]
    
    try:
        # If using AI-based classification
        if query_classifier_mode != "regex" and 'query_classifier' in globals() and query_classifier is not None:
            # Use our AI-powered query classifier
            return query_classifier.classify(text)
        else:
            # Fall back to the regex-based approach with model size
            from utils.query_optimization import categorize_query
            return categorize_query(text, model_size=model_size)
    except Exception as e:
        print(f"Error in query classification: {e}")
        print("Falling back to default classification")
        # Always default to document query on error
        return "DOCUMENT_QUERY", text

def handle_conversation(text):
    """
    Generate a response for casual conversation without using the LLM.
    
    Args:
        text (str): The user's text input
        
    Returns:
        str: A response based on conversation mode and query complexity
    """
    global conversation_mode
    # Pass the query for complexity-aware responses
    return get_conversation_response(conversation_mode, query=text)

@app.post("/ask")
async def ask(request: Request):
    start_time = time.time()  # Track request time
    
    body = await request.json()
    text = body.get("text", "")
    print(f"Received question: {text}")
    
    # Get model capabilities for context-aware processing
    model_capabilities = llm_manager.get_capabilities()
    print(f"Using model size: {model_capabilities['model_size']}")
    
    # Determine if this is a search query, document query or just conversation
    query_type, query = process_query(text, model_capabilities)
    
    if query_type == "DOCUMENT_SEARCH":
        print("Processing as a document search query")
        if query in query_cache:
            print("Found in query cache")
            results = query_cache.get(query)
        else:
            # Use model-specific document limit for search
            k = model_capabilities["document_limit"]
            print(f"Retrieving {k} documents based on model capabilities")
            results = vectorstore.similarity_search(query, k=k)
            query_cache.set(query, results)
        
        content = f"Found {len(results)} relevant documents for: '{query}'\n\nSnippets:\n"
        for doc in results:
            content += f"- {doc.metadata['source']}: {doc.page_content[:200]}...\n"
        
        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds")
        return {"text": content, "sources": [doc.metadata["source"] for doc in results]}
    
    elif query_type == "CONVERSATION":
        print("Query classified as: CONVERSATION")
        # Assess query complexity
        query_complexity = assess_query_complexity(query) if 'assess_query_complexity' in globals() else 1
        print(f"Query complexity: {query_complexity}")
        
        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds")
        return {"text": handle_conversation(query), "sources": []}
    
    elif query_type == "CONCISE_QUERY":
        print("Query classified as: CONCISE_QUERY")
        print("Processing as concise RAG query")
        
        # Check if we have cached document retrieval results
        if query in query_cache:
            print("Using cached document retrieval results")
            docs = query_cache.get(query)
        else:
            # Use model-specific document limit for concise queries
            k = model_capabilities["concise_limit"]
            print(f"Retrieving {k} documents based on model capabilities")
            docs = vectorstore.similarity_search(query, k=k)
            query_cache.set(query, docs)
        
        print(f"Found {len(docs)} relevant documents")
        
        # Optimize context for model size
        docs = optimize_context_for_model(docs, query, model_capabilities)
        
        # Get model-specific prompt templates
        model_prompts = get_model_specific_prompts(model_capabilities)
        
        # Generate answer using the retrieved documents and the question
        from langchain.chains.question_answering import load_qa_chain
        concise_chain = load_qa_chain(llm, chain_type="stuff", prompt=model_prompts["concise"])
        
        answer = concise_chain.invoke({
            "input_documents": docs,
            "query": query
        })
        
        # Apply model-specific formatting and quality control
        formatted_response = format_response(answer["output_text"], model_capabilities)
        
        # Validate response quality (for small models)
        validated_response = validate_response_quality(formatted_response, query, model_capabilities)
        
        # Create hybrid response if needed
        final_response = create_hybrid_response(validated_response, query, docs, model_capabilities)
        
        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds")
        return {"text": final_response, "sources": [doc.metadata["source"] for doc in docs]}
    
    else:  # DOCUMENT_QUERY
        print("Query classified as: DOCUMENT_QUERY")
        print("Processing as RAG query")
        
        # Check if we have cached document retrieval results
        if query in query_cache:
            print("Using cached document retrieval results")
            docs = query_cache.get(query)
        else:
            # Use model-specific document limit for regular queries
            k = model_capabilities["document_limit"]
            print(f"Retrieving {k} documents based on model capabilities")
            docs = vectorstore.similarity_search(query, k=k)
            query_cache.set(query, docs)
        
        print(f"Found {len(docs)} relevant documents")
        
        # Optimize context for model size
        docs = optimize_context_for_model(docs, query, model_capabilities)
        
        # Get model-specific prompt templates
        model_prompts = get_model_specific_prompts(model_capabilities)
        
        # For small models, use model-specific QA chain
        if model_capabilities["model_size"] == "small":
            # Create custom chain with model-specific prompt
            from langchain.chains.question_answering import load_qa_chain
            custom_chain = load_qa_chain(llm, chain_type="stuff", prompt=model_prompts["document"])
            
            answer = custom_chain.invoke({
                "input_documents": docs,
                "query": query
            })
        else:
            # Use pre-initialized QA chain for large models
            answer = qa_chain.invoke({
                "input_documents": docs,
                "query": query
            })
        
        # Apply model-specific formatting and quality control
        formatted_response = format_response(answer["output_text"], model_capabilities)
        
        # Validate response quality (for small models)
        validated_response = validate_response_quality(formatted_response, query, model_capabilities)
        
        # Create hybrid response if needed
        final_response = create_hybrid_response(validated_response, query, docs, model_capabilities)
        
        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds")
        return {"text": final_response, "sources": [doc.metadata["source"] for doc in docs]}

@app.get("/documents")
async def list_documents():
    """List all documents in the vector store."""
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

@app.get("/stats")
async def get_stats():
    """Get statistics about the document processing system."""
    global query_classifier
    
    try:
        # Create a document tracker to get document statistics
        from utils.document_tracker import DocumentTracker
        tracker = DocumentTracker()
        
        # Get cache statistics
        cache_stats = query_cache.get_stats()
        
        # Get document statistics
        doc_count = tracker.get_document_count()
        extensions = tracker.get_file_extensions()
        last_update = tracker.get_last_update()
        
        # Get vector store statistics
        vs_stats = {}
        if vectorstore:
            vs_data = vectorstore.get()
            vs_stats = {
                "total_chunks": len(vs_data["ids"]),
                "embedding_dim": len(vs_data["embeddings"][0]) if vs_data["embeddings"] else 0,
            }
            
        # Get query classifier statistics
        classifier_stats = {}
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

@app.post("/clear-cache")
async def clear_cache():
    """Clear the query cache."""
    try:
        query_cache.clear()
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/clear-classifier-cache")
async def clear_classifier_cache():
    """Clear the query classifier cache."""
    global query_classifier
    
    try:
        if query_classifier is not None:
            query_classifier.clear_cache()
            return {"status": "success", "message": "Query classifier cache cleared successfully"}
        else:
            return {"status": "warning", "message": "No query classifier initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing classifier cache: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global query_classifier, query_classifier_mode, embeddings
    
    try:
        # First initialize vector store (this should always succeed)
        initialize_vectorstore()
        
        # Initialize the query classifier if not using regex mode
        if query_classifier_mode != "regex":
            try:
                from utils.query_classification import QueryClassifier
                print(f"Initializing query classifier in {query_classifier_mode} mode...")
                query_classifier = QueryClassifier(embeddings, classifier_type=query_classifier_mode)
                query_classifier.initialize()
                print("Query classifier initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize query classifier: {e}")
                print("Falling back to regex-based classification")
                query_classifier_mode = "regex"
                query_classifier = None
        
        # Then try to initialize the LLM (this might fail, but that's okay)
        try:
            initialize_llm()
            initialize_qa_chain()
        except Exception as e:
            print(f"Warning: LLM initialization failed, but application will continue: {e}")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global vectorstore, llm, qa_chain, query_classifier
    
    print("Shutting down and cleaning up resources...")
    
    # Clean up query classifier
    if query_classifier is not None:
        try:
            query_classifier.clear_cache()
            print("Query classifier cache cleared")
        except Exception as e:
            print(f"Warning: Error cleaning up query classifier: {e}")
    
    # Clean up vector store
    if vectorstore is not None:
        try:
            # Close any open connections if possible
            if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
                vectorstore._client.close()
        except Exception as e:
            print(f"Warning: Error cleaning up vector store: {e}")
    
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
    vectorstore = None
    llm = None
    qa_chain = None
    query_classifier = None
    
    print("Cleanup complete")

@app.get("/config")
async def get_config():
    """
    Get current system configuration.
    
    Returns:
        Dict: Current configuration values
    """
    global max_context_tokens, conversation_mode, query_classifier_mode
    
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

def format_response(response_text, model_capabilities):
    """
    Apply model-specific formatting to ensure quality responses.
    
    Args:
        response_text (str): Raw response from the model
        model_capabilities (dict): Model capabilities dictionary
        
    Returns:
        str: Formatted response text
    """
    import re
    
    model_size = model_capabilities["model_size"]
    
    # For small models, apply more aggressive formatting and checks
    if model_size == "small":
        # Remove potential hallucinations indicated by uncertain language
        response_text = re.sub(r'(?i)\b(I\'m not sure|I don\'t know|I believe|probably|might be|possibly)\b',
                              '', response_text)
        
        # Ensure the response isn't too long for small models
        words = response_text.split()
        if len(words) > 100:
            response_text = ' '.join(words[:100]) + '...'
    
    # For all models, clean up formatting
    response_text = response_text.strip()
    
    # Remove any extra newlines (more than 2 consecutive)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    # Remove repeated spaces
    response_text = re.sub(r' {2,}', ' ', response_text)
    
    # Ensure sentence capitalization
    sentences = re.split(r'(?<=[.!?])\s+', response_text)
    for i, sentence in enumerate(sentences):
        if sentence and sentence[0].islower():
            sentences[i] = sentence[0].upper() + sentence[1:]
    
    response_text = ' '.join(sentences)
    
    return response_text

def validate_response_quality(response, query, model_capabilities):
    """
    Validate the quality of the model's response.
    Returns the original response if good, or a fallback if poor quality.
    
    Args:
        response (str): The model's response
        query (str): The original query
        model_capabilities (dict): Model capabilities dictionary
        
    Returns:
        str: Original response or fallback text
    """
    model_size = model_capabilities["model_size"]
    
    # Only apply strict validation to small models
    if model_size != "small":
        return response
    
    # Check for potential low-quality indicators
    low_quality = False
    
    # 1. Too short responses
    if len(response.split()) < 5:
        print("Quality issue: Response too short")
        low_quality = True
    
    # 2. Repetitive content
    words = response.lower().split()
    unique_words = set(words)
    if len(words) > 0 and len(unique_words) / len(words) < 0.5:
        print("Quality issue: Repetitive content")
        low_quality = True
    
    # 3. Nonsensical or incomplete sentences
    if response.count('.') == 0 or not response.strip().endswith(('.', '?', '!')):
        print("Quality issue: Incomplete or missing punctuation")
        low_quality = True
    
    # 4. Response doesn't seem to address the query
    query_keywords = set([w.lower() for w in query.split() if len(w) > 3])
    response_text = response.lower()
    matches = sum(1 for word in query_keywords if word in response_text)
    if len(query_keywords) > 0 and matches / len(query_keywords) < 0.2:
        print("Quality issue: Response doesn't address the query")
        low_quality = True
    
    # If low quality detected, provide a fallback response
    if low_quality:
        return generate_fallback_response(query)
    
    return response

def generate_fallback_response(query):
    """
    Generate a fallback response when the model output is low quality.
    
    Args:
        query (str): The original query
        
    Returns:
        str: Fallback response text
    """
    query_type = "information"
    
    # Determine query type for better fallback response
    if any(word in query.lower() for word in ["how", "process", "steps", "instruction"]):
        query_type = "process"
    elif any(word in query.lower() for word in ["what is", "definition", "meaning", "define"]):
        query_type = "definition"
    elif any(word in query.lower() for word in ["example", "show me", "instance"]):
        query_type = "example"
    
    # Create fallback responses by type
    fallbacks = {
        "process": (
            "Based on the documentation, I can't provide a complete step-by-step explanation. "
            "You might want to look at specific implementation details in the codebase for this process."
        ),
        "definition": (
            "The documentation doesn't provide a clear definition for this term. "
            "You might want to search for specific usage examples in the codebase."
        ),
        "example": (
            "I couldn't find a good example of this in the available documentation. "
            "Try looking for test cases or implementation files to see practical examples."
        ),
        "information": (
            "Based on the documentation, I can't provide a complete answer to this question. "
            "The available information is limited, but you might find relevant details by "
            "searching for specific keywords related to your question."
        )
    }
    
    return fallbacks.get(query_type, fallbacks["information"])

def optimize_context_for_model(docs, query, model_capabilities):
    """
    Optimize document context based on model capabilities.
    
    Args:
        docs (list): List of Document objects
        query (str): The query string
        model_capabilities (dict): Dictionary containing model capabilities
        
    Returns:
        list: Optimized list of Document objects
    """
    from langchain.schema import Document
    
    if not docs:
        return []
    
    optimized_docs = []
    model_size = model_capabilities.get("model_size", "small")
    context_window = model_capabilities.get("context_window", 2048)
    
    if model_size == "small":
        # Calculate approximate max tokens per document
        # Assume we need about 25% of context window for query and response
        max_tokens = int((context_window * 0.75) / max(1, len(docs)))
        
        # For small models, always use testing limit of 1536 characters
        char_limit = 1536
        
        print(f"Small model: limiting each document to ~{max_tokens} tokens ({char_limit} chars)")
        
        # Truncate document content to fit
        for i, doc in enumerate(docs):
            content = doc.page_content
            print(f"Doc {i+1} original length: {len(content)} chars, limit: {char_limit} chars")
            
            if len(content) > char_limit:
                # Prioritize beginning of documents
                new_content = content[:char_limit] + "..."
                print(f"Truncated document {i+1} from {len(content)} chars to {len(new_content)} chars")
            else:
                new_content = content
                print(f"Document {i+1} within limit, keeping original content")
                
            # Verify content length
            print(f"Final content length for doc {i+1}: {len(new_content)} chars")
                
            # Create a new document with the optimized content
            new_doc = Document(
                page_content=new_content,
                metadata=doc.metadata.copy() if doc.metadata else {}  # Make a copy of the metadata
            )
            optimized_docs.append(new_doc)
    else:
        # For larger models, we can be more generous but still optimize
        max_tokens = int((context_window * 0.85) / max(1, len(docs)))
        char_limit = max_tokens * 4  # Rough estimate: 1 token ≈ 4 characters
        
        # Ensure a reasonable limit (for testing with large values)
        if char_limit >= 8000:
            char_limit = 3072  # Force a smaller limit for testing
            
        print(f"Large model: limiting each document to ~{max_tokens} tokens ({char_limit} chars)")
        
        # More balanced truncation for larger models
        for i, doc in enumerate(docs):
            content = doc.page_content
            print(f"Doc {i+1} original length: {len(content)} chars, limit: {char_limit} chars")
            
            if len(content) > char_limit:
                # Keep both beginning and end as these may have important info
                half_length = int(char_limit / 2)
                if half_length * 2 + 3 >= len(content):
                    # If almost fits, just keep it all
                    new_content = content
                    print(f"Document {i+1} almost fits, keeping original content")
                else:
                    new_content = content[:half_length] + "..." + content[-half_length:]
                    print(f"Balanced truncation of document {i+1} from {len(content)} chars to {len(new_content)} chars")
            else:
                new_content = content
                print(f"Document {i+1} within limit, keeping original content")
                
            # Verify content length
            print(f"Final content length for doc {i+1}: {len(new_content)} chars")
                
            # Create a new document with the optimized content
            new_doc = Document(
                page_content=new_content,
                metadata=doc.metadata.copy() if doc.metadata else {}  # Make a copy of the metadata
            )
            optimized_docs.append(new_doc)
    
    # Final verification
    for i, doc in enumerate(optimized_docs):
        print(f"Optimized doc {i+1} final length: {len(doc.page_content)} chars")
    
    # Return the new document list
    return optimized_docs

def create_hybrid_response(llm_response, query, docs, model_capabilities):
    """
    Create a hybrid response combining template text with LLM output.
    Used when LLM response might be low quality.
    
    Args:
        llm_response (str): The raw LLM-generated response
        query (str): The original query
        docs (list): List of documents used for context
        model_capabilities (dict): Model capabilities dictionary
        
    Returns:
        str: Original or hybrid response text
    """
    model_size = model_capabilities["model_size"]
    
    # Only apply for small models
    if model_size != "small":
        return llm_response
    
    # Check confidence in the response
    confidence = assess_response_confidence(llm_response, query)
    print(f"Response confidence score: {confidence:.2f}")
    
    if confidence < 0.4:  # Low confidence threshold
        # Create a templated response with the LLM output incorporated
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        source_str = ", ".join(sources[:2])
        if len(sources) > 2:
            source_str += f" and {len(sources)-2} more"
            
        # Hybrid response template
        hybrid = (
            f"Based on information from {source_str}, I found: "
            f"{llm_response}\n\n"
            f"Note: This information is summarized from the documentation and may be incomplete."
        )
        return hybrid
    
    return llm_response

def assess_response_confidence(response, query):
    """
    Assess confidence level in the model's response.
    Returns a score between 0-1 (higher is better).
    
    Args:
        response (str): The model's response
        query (str): The original query
        
    Returns:
        float: Confidence score from 0-1
    """
    # Simple heuristics for confidence assessment
    confidence = 0.5  # Start with neutral confidence
    
    # Length-based assessment (too short or too long are suspicious)
    words = response.split()
    if len(words) < 10:
        confidence -= 0.2
    elif len(words) > 200:
        confidence -= 0.1
    
    # Check for hedging language
    hedging_terms = ['i think', 'probably', 'might', 'may be', 'possibly', 
                     'i believe', 'perhaps', 'seems', 'could be']
    hedges = sum(1 for term in hedging_terms if term in response.lower())
    confidence -= 0.1 * min(3, hedges)  # Max penalty of 0.3
    
    # Check for query term presence
    query_terms = [term.lower() for term in query.split() if len(term) > 3]
    matched = sum(1 for term in query_terms if term in response.lower())
    if query_terms:
        term_match_ratio = matched / len(query_terms)
        confidence += 0.2 * term_match_ratio  # Max bonus of 0.2
    
    # Check for definitive language
    definitive_terms = ['is', 'are', 'was', 'were', 'definitely', 'certainly', 'always', 'never']
    definitive_count = sum(1 for term in definitive_terms if f" {term} " in f" {response.lower()} ")
    confidence += 0.05 * min(3, definitive_count)  # Small bonus for confident language
    
    # Ensure confidence is within 0-1 range
    return max(0.0, min(1.0, confidence))

if __name__ == "__main__":
    import uvicorn
    
    # Initialize the vector store (this should always work)
    print("Initializing vector store...")
    initialize_vectorstore()
    
    # Initialize the query classifier
    if query_classifier_mode != "regex":
        try:
            from utils.query_classification import QueryClassifier
            print(f"Initializing query classifier in {query_classifier_mode} mode...")
            query_classifier = QueryClassifier(embeddings, classifier_type=query_classifier_mode)
            query_classifier.initialize()
            print("Query classifier initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize query classifier: {e}")
            print("Falling back to regex-based classification")
            query_classifier_mode = "regex"
            query_classifier = None
    
    # Try to initialize the LLM but continue if it fails
    print("Initializing LLM (optional)...")
    try:
        if initialize_llm():
            initialize_qa_chain()
        else:
            print("Application will run without LLM capabilities")
    except Exception as e:
        print(f"Warning: LLM initialization failed: {e}")
        print("Application will run without LLM capabilities")
    
    print("System initialization complete, starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 