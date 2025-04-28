"""
MindNest - AI Documentation System
Main application entry point
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request

# Initialize environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import core components
from mindnest.core.llm_manager import initialize_llm, initialize_qa_chain
from mindnest.core.document_processor import initialize_vectorstore
from mindnest.api.endpoints import router as api_router
from mindnest.core.config import initialize_query_classifier, query_classifier_mode

# Create FastAPI application
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

# Include API router
app.include_router(api_router)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    try:
        # First initialize vector store (this should always succeed)
        initialize_vectorstore()
        
        # Initialize the query classifier
        initialize_query_classifier()
        
        # Try to initialize the LLM (this might fail, but that's okay)
        try:
            llm_success = initialize_llm()
            if llm_success:
                print("LLM initialization successful!")
                qa_success = initialize_qa_chain()
                if qa_success:
                    print("QA chain initialization successful!")
                else:
                    print("Warning: QA chain initialization failed, but application will continue")
            else:
                print("Warning: LLM initialization failed, but application will continue")
        except Exception as e:
            print(f"Warning: LLM initialization failed, but application will continue: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    from mindnest.core.llm_manager import cleanup_resources
    cleanup_resources()
    print("Cleanup complete")

def run_server(host="0.0.0.0", port=8000):
    """Run the server directly."""
    print("System initialization complete, starting server...")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Initialize the vector store (this should always work)
    print("Initializing vector store...")
    initialize_vectorstore()
    
    # Initialize the query classifier
    initialize_query_classifier()
    
    # Try to initialize the LLM but continue if it fails
    print("Initializing LLM (optional)...")
    try:
        llm_success = initialize_llm()
        if llm_success:
            print("LLM initialized successfully!")
            qa_success = initialize_qa_chain()
            if qa_success:
                print("QA chain initialized successfully!")
            else:
                print("Warning: QA chain initialization failed")
                print("Application will run with limited capabilities")
        else:
            print("Warning: LLM initialization failed")
            print("Application will run with limited capabilities")
    except Exception as e:
        print(f"Warning: LLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("Application will run with limited capabilities")
    
    run_server() 