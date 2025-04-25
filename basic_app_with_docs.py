import os
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    return {"status": "healthy", "model": model_name}

# Initialize the model with absolute path
model_name = "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf"
model_path = os.path.abspath(f"models/{model_name}")
print(f"Loading model from: {model_path}")

try:
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
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Document storage
documents_storage = []
document_chunks = []

def process_document(file_path: str) -> List[Document]:
    """Process a document and split it into chunks."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Choose appropriate loader based on file extension
        if file_extension == '.pdf':
            loader = PDFMinerLoader(file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:  # Default to text loader for .txt and other files
            loader = TextLoader(file_path)
        
        # Load the document
        doc = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(doc)
        
        print(f"Processed {file_path}, generated {len(chunks)} chunks")
        return chunks
    
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return []

def simple_keyword_search(query: str, docs: List[Document], max_results: int = 3) -> List[Document]:
    """
    Simple keyword-based search that doesn't require embeddings.
    Returns documents containing words from the query.
    """
    if not docs:
        return []
    
    # Extract unique words from the query, excluding common words
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "as", "of", "and", "or", "is", "are"}
    query_words = set([word.lower() for word in re.findall(r'\b\w+\b', query) if word.lower() not in stopwords])
    
    # Assign a relevance score to each document based on keyword matches
    scored_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        score = sum(1 for word in query_words if word in content)
        if score > 0:  # Only include documents with at least one matching keyword
            scored_docs.append((score, doc))
    
    # Sort by relevance score (highest first) and take top results
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs[:max_results]]

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str
    sources: List[str] = []

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int

# Upload endpoint for adding documents
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Create a docs directory if it doesn't exist
        os.makedirs("docs", exist_ok=True)
        
        # Save the file
        file_path = os.path.join("docs", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process the document
        chunks = process_document(file_path)
        
        # Store document info
        documents_storage.append({
            "filename": file.filename,
            "path": file_path
        })
        
        # Store document chunks
        document_chunks.extend(chunks)
        
        return {"message": f"Successfully processed {file.filename}", "chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

# List documents endpoint
@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    result = []
    for doc in documents_storage:
        # Count chunks for this document
        chunk_count = sum(1 for chunk in document_chunks if chunk.metadata.get("source") == doc["path"])
        result.append(DocumentInfo(filename=doc["filename"], chunk_count=chunk_count))
    
    return result

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Handle incoming questions and provide answers with document context."""
    try:
        # Search for relevant documents
        relevant_docs = simple_keyword_search(question.text, document_chunks)
        sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
        
        # Format the context from relevant documents
        context = ""
        if relevant_docs:
            formatted_docs = []
            for i, doc in enumerate(relevant_docs):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                formatted_docs.append(f"Document {i+1} ({source}):\n{doc.page_content}")
            context = "\n\n".join(formatted_docs)
            context = f"\nHere are some relevant documents that might help:\n\n{context}\n\n"
        
        # Format the prompt with context
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.{context}

USER: {question.text}
ASSISTANT:"""
        
        # Get answer from LLM
        try:
            # Try the new method first
            response = llm.invoke(prompt)
        except (AttributeError, TypeError):
            # Fall back to the old method if invoke doesn't work
            response = llm(prompt)
        
        # Print the generated answer for debugging (just first 100 chars to avoid console clutter)
        print(f"Generated answer: {response[:100]}...")
        
        # Create response with sources
        source_paths = [os.path.basename(src) for src in sources] if sources else []
        return Answer(text=response, sources=source_paths)
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 