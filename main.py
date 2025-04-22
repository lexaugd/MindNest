import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PythonLoader
)
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# Initialize the model and embeddings
model_path = os.getenv("MODEL_PATH", "models/llama-2-7b.Q4_K_M.gguf")
try:
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.3,  # Lower temperature for more focused responses
        max_tokens=2000,
        top_p=0.95,  # Slightly lower top_p for more focused sampling
        verbose=True,
        n_ctx=2048,
        n_batch=512,
        repeat_penalty=1.1  # Add repeat penalty to reduce repetition
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the vector store
vectorstore = None

def load_documents():
    """Load and process all documents from the docs directory."""
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
            loader = loader_class("docs", **loader_args)
            # Process files in batches to manage memory
            batch_size = 10
            for i in range(0, len(loader.file_paths), batch_size):
                batch_paths = loader.file_paths[i:i + batch_size]
                batch_docs = loader.load(batch_paths)
                documents.extend(batch_docs)
                # Clear memory after each batch
                del batch_docs
        except Exception as e:
            print(f"Error loading {ext} files: {e}")
    
    # Optimize chunk size based on document count
    total_docs = len(documents)
    if total_docs > 50:
        chunk_size = 800  # Smaller chunks for large collections
        chunk_overlap = 100
    else:
        chunk_size = 1000
        chunk_overlap = 200
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)

def initialize_vectorstore():
    """Initialize the vector store with documents."""
    global vectorstore
    try:
        documents = load_documents()
        # Use persistent storage for large collections
        persist_directory = "chroma_db"
        if os.path.exists(persist_directory):
            # Load existing vector store
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        else:
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            # Persist the vector store
            vectorstore.persist()
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

# Initialize the conversation chain
qa_chain = None

def initialize_qa_chain():
    """Initialize the question-answering chain."""
    global qa_chain
    if vectorstore is None:
        initialize_vectorstore()
    
    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        # Create a custom prompt template
        template = """You are a code assistant that answers questions about the provided code. 
        Follow these rules strictly:
        1. Only answer based on the actual code shown in the context
        2. Be concise and direct - no explanations unless asked
        3. If the code doesn't contain the information needed, say "I don't have enough information to answer that question"
        4. Answer only the specific question asked, do not include additional questions or answers
        5. If you see multiple code blocks separated by "---", treat them as separate files
        6. When referring to code, specify which file/block you're referring to if there are multiple
        7. Do not make assumptions about code that isn't shown
        8. Do not generate example code unless specifically asked
        9. If asked about classes, list them with their key features (fields, methods)
        10. Keep responses focused and to the point
        11. When listing methods, include ALL methods including toString() and constructors
        12. Always specify which file you're referring to in your answer
        
        Here is the code:
        {context}
        
        Question: {question}
        
        Answer:"""
        
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
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        raise

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str
    sources: List[str]

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Handle incoming questions and return answers."""
    if qa_chain is None:
        initialize_qa_chain()
    
    try:
        # Optimize similarity search based on collection size
        k = 2  # Default number of documents to retrieve
        if vectorstore is not None:
            collection_size = len(vectorstore.get()['ids'])
            if collection_size > 50:
                k = 1  # Retrieve fewer documents for large collections
        
        # Get relevant documents
        docs = vectorstore.similarity_search(question.text, k=k)
        
        # Deduplicate content and ensure proper formatting
        seen_content = set()
        unique_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                # Add file type and path indicator
                file_path = doc.metadata.get("source", "Unknown")
                file_type = file_path.split(".")[-1]
                content = f"File: {file_path}\nType: {file_type}\n{content}"
                unique_docs.append(doc)
        
        # Join unique documents with clear separation
        context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
        sources = [doc.metadata.get("source", "Unknown") for doc in unique_docs]
        
        # Get answer from LLM
        result = qa_chain.invoke({
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
        
        # Create response
        response = Answer(text=answer, sources=sources)
        return response
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    try:
        initialize_vectorstore()
        initialize_qa_chain()
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 