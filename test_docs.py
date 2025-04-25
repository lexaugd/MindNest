import os
import shutil
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PythonLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Try to use HuggingFaceEmbeddings, fall back to fake embeddings if needed
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
    try:
        print("Initializing vector store...")
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
                return None
                
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
        
        return vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

def test_search(vectorstore, query):
    """Test the vector search with a given query."""
    if vectorstore is None:
        print("No vector store available for search")
        return
    
    print(f"\nSearching for: '{query}'")
    k = 3  # Number of results to retrieve
    docs = vectorstore.similarity_search(query, k=k)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        print(f"\nResult {i+1} from {source}:")
        print("-" * 40)
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        print("-" * 40)

if __name__ == "__main__":
    print("Starting document processing test...")
    vectorstore = initialize_vectorstore()
    
    if vectorstore is not None:
        # Test some queries
        test_search(vectorstore, "What is a TestGroovy class?")
        test_search(vectorstore, "Java class features")
        test_search(vectorstore, "Documentation examples")
    
    print("Test complete!") 