import os
import sys
import uvicorn
from main import app, initialize_vectorstore

# Make sure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    # Initialize just the vector store (skip LLM)
    print("Initializing vector store only...")
    initialize_vectorstore()
    
    # Run the FastAPI server
    print("Starting server with vector search capabilities only...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 