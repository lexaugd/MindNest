import os
from typing import List
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

# Import our optimized components
from utils.incremental_vectorstore import IncrementalVectorStore
from utils.query_cache import QueryCache

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
    return {
        "status": "healthy", 
        "llm": llm_status,
        "vectorstore": vectorstore_status,
        "model": model_name
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
model_name = "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf"
query_cache = QueryCache()  # Initialize query cache

# This function retains backward compatibility but is now just a wrapper
def load_documents():
    """Load and process all documents from the docs directory."""
    print("Loading documents using incremental vector store implementation...")
    inc_vectorstore = IncrementalVectorStore()
    # Force rebuild is handled inside initialize_or_update
    vs = inc_vectorstore.initialize_or_update(force_rebuild=False)
    return []  # Return empty list since the vector store is already populated

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
    global llm, model_name
    
    try:
        from langchain_community.llms import LlamaCpp
        
        print("Initializing LLM...")
        model_path = os.path.abspath(f"models/{model_name}")
        print(f"Loading model from: {model_path}")
        
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.3,
            max_tokens=2000,
            top_p=0.95,
            verbose=True,
            n_ctx=4096,
            n_batch=512,
            repeat_penalty=1.1,
            f16_kv=True
        )
        print("LLM initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return False

def initialize_qa_chain():
    """Initialize the question-answering chain with the LLM."""
    global qa_chain, llm
    
    try:
        if llm is None:
            # If LLM initialization failed, we can't create the QA chain
            print("LLM not initialized. QA chain initialization skipped.")
            return False
    
        print("Initializing QA chain...")
        
        # Create QA chain with proper handling for document retrieval
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        # Define prompt template for retrieval QA
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Create retriever from vectorstore
        retriever = vectorstore.as_retriever()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        print("QA chain initialized successfully")
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

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Process a question using retrieval-augmented generation.
    
    Args:
        question: The question to process
        
    Returns:
        Answer: The generated answer with source documents
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain not initialized. Service unavailable.")
    
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Service unavailable.")
        
    query = question.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    print(f"Received question: {query}")
    
    try:
        # Check if this is a simple query to the documents (no need for LLM)
        if query.lower().startswith("find ") or query.lower().startswith("search "):
            print("Processing as a document search query")
            
            # Extract the actual search query
            search_query = query[5:].strip() if query.lower().startswith("find ") else query[7:].strip()
            
            # Try to get cached results first
            cached_results = query_cache.get(search_query, k=4)
            
            if cached_results:
                print("Found in query cache")
                # Format the result
                sources = []
                
                for result in cached_results:
                    source = result.get("metadata", {}).get("source", "Unknown")
                    if source not in sources:
                        sources.append(source)
                    
                return Answer(
                    text=f"Found {len(cached_results)} relevant documents for: '{search_query}'",
                    sources=sources
                )
            
            # If not in cache, perform vector search
            docs = vectorstore.similarity_search(search_query, k=4)
            
            # Cache the results for future searches
            cacheable_results = []
            for doc in docs:
                cacheable_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            if cacheable_results:
                query_cache.set(search_query, cacheable_results, k=4)
            
            # Format the result
            sources = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
                
            return Answer(
                text=f"Found {len(docs)} relevant documents for: '{search_query}'",
                sources=sources
            )
            
        # Process as a regular RAG query
        print("Processing as RAG query")
        
        # Try to get cached results for document retrieval
        cached_docs = query_cache.get(f"docs:{query}", k=4)
        
        if cached_docs:
            print("Using cached document retrieval results")
            docs = []
            for doc_dict in cached_docs:
                from langchain.docstore.document import Document
                doc = Document(
                    page_content=doc_dict.get("content", ""),
                    metadata=doc_dict.get("metadata", {})
                )
                docs.append(doc)
        else:
            # Get relevant documents from the vector store
            docs = vectorstore.similarity_search(query, k=4)
            
            # Cache the document retrieval results
            cacheable_docs = []
            for doc in docs:
                cacheable_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            if cacheable_docs:
                query_cache.set(f"docs:{query}", cacheable_docs, k=4)
        
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
                
        print(f"Found {len(docs)} relevant documents")
                
        # Generate answer using the retrieved documents and question
        result = qa_chain.invoke({
            "question": query,
            "input_documents": docs,
        })
        
        answer_text = result.get("answer", result.get("result", "No answer generated."))
            
        return Answer(
            text=answer_text,
            sources=sources
        )
                
    except Exception as e:
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

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
            "vectorstore": vs_stats
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

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    try:
        # First initialize vector store (this should always succeed)
        initialize_vectorstore()
        
        # Then try to initialize the LLM (this might fail, but that's okay)
        try:
            initialize_llm()
            initialize_qa_chain()
        except Exception as e:
            print(f"Warning: LLM initialization failed, but application will continue: {e}")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    
    # Initialize the vector store (this should always work)
    print("Initializing vector store...")
    initialize_vectorstore()
    
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