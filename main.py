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

def load_documents():
    """Load and process all documents from the docs directory."""
    print("Loading documents from docs directory...")
    loaders = {
        ".txt": (DirectoryLoader, {"glob": "**/*.txt", "loader_cls": TextLoader}),
        ".md": (DirectoryLoader, {"glob": "**/*.md", "loader_cls": UnstructuredMarkdownLoader}),
        ".py": (DirectoryLoader, {"glob": "**/*.py", "loader_cls": PythonLoader}),
        ".groovy": (DirectoryLoader, {"glob": "**/*.groovy", "loader_cls": TextLoader}),
        ".java": (DirectoryLoader, {"glob": "**/*.java", "loader_cls": TextLoader}),
        ".js": (DirectoryLoader, {"glob": "**/*.js", "loader_cls": TextLoader}),
        ".ts": (DirectoryLoader, {"glob": "**/*.ts", "loader_cls": TextLoader}),
    }
    
    documents = []
    for ext, (loader_class, loader_args) in loaders.items():
        try:
            # Create and load documents directly without trying to access file_paths first
            loader = loader_class("docs", **loader_args)
            try:
                ext_docs = loader.load()
                if ext_docs:
                    print(f"Loaded {len(ext_docs)} {ext} files")
                    documents.extend(ext_docs)
                else:
                    print(f"No {ext} files found in docs directory")
            except Exception as e:
                print(f"Error loading {ext} files: {e}")
        except Exception as e:
            print(f"Error creating loader for {ext} files: {e}")
    
    if not documents:
        print("No documents found in docs directory!")
        return []
        
    # Optimize chunk size based on document count
    total_docs = len(documents)
    print(f"Total documents loaded: {total_docs}")
    if total_docs > 50:
        chunk_size = 800  # Smaller chunks for large collections
        chunk_overlap = 100
    else:
        chunk_size = 1000
        chunk_overlap = 200
    
    print(f"Splitting documents with chunk size {chunk_size} and overlap {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Document splitting complete. Created {len(split_docs)} chunks")
    return split_docs

def initialize_vectorstore():
    """Initialize the vector store with documents."""
    global vectorstore
    try:
        print("Initializing vector store...")
        # Check if there's existing data
        persist_directory = "chroma_db"
        
        # Option to force rebuild the vector store
        force_rebuild = False
        
        if force_rebuild and os.path.exists(persist_directory):
            print("Removing existing vector store for rebuild")
            shutil.rmtree(persist_directory)
        
        if os.path.exists(persist_directory):
            # Load existing vector store
            print("Loading existing vector store")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print(f"Loaded existing vector store with {len(vectorstore.get()['ids'])} documents")
        else:
            # Create new vector store
            print("Creating new vector store")
            documents = load_documents()
            if not documents:
                print("No documents to add to vector store!")
                return
                
            print(f"Adding {len(documents)} documents to vector store")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            # Persist the vector store
            print("Persisting vector store")
            vectorstore.persist()
            print("Vector store created and persisted successfully")
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
    """Initialize the question-answering chain."""
    global qa_chain, llm
    
    if vectorstore is None:
        initialize_vectorstore()
    
    if llm is None:
        if not initialize_llm():
            print("Cannot initialize QA chain without LLM")
            return False
    
    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        print("Initializing QA chain")
        # Create a custom prompt template with Vicuna format
        template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

The assistant is presented with code or documentation from different files. The assistant's task is to:
1. Only answer based on the actual content shown in the context
2. Be concise and direct - no explanations unless asked
3. If the content doesn't contain the information needed, say "I don't have enough information to answer that question"
4. Answer only the specific question asked, do not include additional questions or answers
5. If you see multiple content blocks separated by "---", treat them as separate files
6. When referring to code or content, specify which file/block you're referring to if there are multiple
7. Do not make assumptions about content that isn't shown
8. Do not generate example code unless specifically asked
9. Keep responses focused and to the point
10. Always specify which file you're referring to in your answer

Here is the content:
{context}

USER: {question}
ASSISTANT:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True
        )
        print("QA chain initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        return False

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str
    sources: List[str]

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Handle incoming questions and return answers."""
    global qa_chain, vectorstore, llm
    
    # Make sure vectorstore is initialized
    if vectorstore is None:
        initialize_vectorstore()
        
    # If vectorstore still None, return error
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not available")
    
    try:
        print(f"Processing question: {question.text}")
        
        # Get relevant documents from vector store
        k = 2  # Default number of documents to retrieve
        collection_size = len(vectorstore.get()['ids'])
        print(f"Collection size: {collection_size} documents")
        if collection_size > 50:
            k = 1  # Retrieve fewer documents for large collections
        
        # Get relevant documents
        print(f"Performing similarity search with k={k}")
        docs = vectorstore.similarity_search(question.text, k=k)
        print(f"Found {len(docs)} relevant documents")
        
        # Deduplicate content and ensure proper formatting
        seen_content = set()
        unique_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                # Add file type and path indicator
                file_path = doc.metadata.get("source", "Unknown")
                file_type = file_path.split(".")[-1] if "." in file_path else "txt"
                content = f"File: {file_path}\nType: {file_type}\n{content}"
                unique_docs.append(doc)
        
        print(f"After deduplication: {len(unique_docs)} unique documents")
        sources = [doc.metadata.get("source", "Unknown") for doc in unique_docs]
        
        # If LLM is not initialized, just return the sources without generating answer
        if llm is None:
            print("LLM not initialized, returning sources only")
            answer = "LLM not available. Here are the relevant documents: " + ", ".join(sources)
            return Answer(text=answer, sources=sources)
        
        # Make sure QA chain is initialized
        if qa_chain is None:
            if not initialize_qa_chain():
                raise HTTPException(status_code=500, detail="QA chain initialization failed")
        
        # Join unique documents with clear separation
        context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
        
        # Get answer from LLM
        print("Generating answer from LLM")
        try:
            # Try the new method first
            result = qa_chain.invoke({
                "context": context,
                "question": question.text
            })
        except (AttributeError, TypeError):
            # Fall back to the old method if invoke doesn't work
            result = qa_chain({
                "context": context,
                "question": question.text
            })
        
        # Extract and validate answer
        answer = result.get("text", "").strip()
        
        # Basic validation of response
        if "I don't have enough information" not in answer:
            # Check if response is too long (more than 500 characters)
            if len(answer) > 500:
                answer = answer[:500] + "... (response truncated)"
            
            # Check if response contains multiple questions
            if "Question:" in answer:
                answer = answer.split("Question:")[0].strip()
            
            # Ensure file reference is included if multiple files are present
            if len(unique_docs) > 1 and "File:" not in answer:
                answer = f"Based on {sources[0]}: {answer}"
        
        # Print the generated answer for debugging
        print(f"Generated answer: {answer[:100]}...")
        
        # Create response
        response = Answer(text=answer, sources=sources)
        return response
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all documents in the vector store."""
    if vectorstore is None:
        initialize_vectorstore()
        
    try:
        if vectorstore is None:
            return {"documents": []}
            
        collection_data = vectorstore.get()
        # Get metadata for all documents
        docs = []
        for i, doc_id in enumerate(collection_data['ids']):
            if i < len(collection_data['metadatas']):
                metadata = collection_data['metadatas'][i]
                source = metadata.get('source', 'Unknown')
                docs.append({"id": doc_id, "source": source})
        
        # Group by source file
        doc_counts = {}
        for doc in docs:
            source = doc['source']
            if source in doc_counts:
                doc_counts[source] += 1
            else:
                doc_counts[source] = 1
                
        result = [{"filename": src, "chunk_count": count} for src, count in doc_counts.items()]
        return {"documents": result, "total_chunks": len(docs)}
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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