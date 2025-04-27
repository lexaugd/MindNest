"""
LLM Manager for MindNest.
Handles initialization and management of language models.
"""

import os
from typing import Optional, Dict, Any

from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from utils.config import config
from utils.logger import get_logger

# Get module logger
logger = get_logger(__name__)

class LLMManager:
    """
    Manages language model initialization and usage.
    Provides centralized access to LLM functionality.
    """
    
    def __init__(self, use_small_model: Optional[bool] = None):
        """
        Initialize the LLM manager.
        
        Args:
            use_small_model: Whether to use the small model (default: from config)
        """
        self.use_small_model = use_small_model if use_small_model is not None else config.use_small_model
        
        # Initialize model to None
        self.llm = None
        self.qa_chain = None
        
        # Track initialization status
        self.initialized = False
        
        # Model configurations
        self.model_path = config.get_model_path(self.use_small_model)
        
        # Configure model settings based on model size
        self.context_window = config.context_window
        self.batch_size = config.batch_size
    
    def initialize(self) -> bool:
        """
        Initialize the language model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.initialized and self.llm is not None:
            logger.info("LLM already initialized")
            return True
            
        try:
            logger.info(f"Initializing LLM using model: {os.path.basename(self.model_path)}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            # Initialize the model
            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                verbose=True,
                n_ctx=self.context_window,
                n_batch=self.batch_size,
                repeat_penalty=1.1,
                f16_kv=True,
                use_mlock=True,
                n_gpu_layers=40,
                seed=42,
                stop=["</s>"]
            )
            
            logger.info(f"LLM initialized successfully with {self.context_window} context window")
            self.initialized = True
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.initialized = False
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the current model.
        
        Returns:
            Dict with model capabilities including:
            - model_size: "small" or "large"
            - context_window: maximum context window size
            - document_limit: recommended number of documents for retrieval
            - concise_limit: recommended number of documents for concise queries
            - token_capacity: approximate token capacity
            - complexity_level: 1-5 scale of complexity the model can handle
        """
        capabilities = {
            "model_size": "small" if self.use_small_model else "large",
            "context_window": self.context_window,
            "token_capacity": 2000 if self.use_small_model else 4000,
            "complexity_level": 3 if self.use_small_model else 5,
        }
        
        # Document retrieval recommendations
        if self.use_small_model:
            capabilities["document_limit"] = 3  # Fewer docs for small models
            capabilities["concise_limit"] = 2   # Even fewer for concise queries
        else:
            capabilities["document_limit"] = 5  # More docs for large models
            capabilities["concise_limit"] = 3   # Fewer but still substantial for concise
        
        logger.debug(f"Model capabilities: {capabilities}")    
        return capabilities
    
    def _initialize_qa_chain(self) -> None:
        """Initialize question-answering chain with the LLM."""
        try:
            if self.llm is None:
                logger.warning("LLM not initialized. QA chain initialization skipped.")
                return
                
            logger.info("Initializing QA chain...")
            
            # Define an improved prompt template that adapts to the question
            template = """
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
            
            prompt = PromptTemplate(
                template=template, 
                input_variables=["context", "query"]
            )
            
            # Create QA chain with our optimized prompt
            self.qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            logger.info("QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
    
    def get_llm(self) -> Optional[LlamaCpp]:
        """
        Get the initialized LLM instance.
        
        Returns:
            Optional[LlamaCpp]: The LLM instance or None if not initialized
        """
        if not self.initialized:
            success = self.initialize()
            if not success:
                return None
                
        return self.llm
    
    def get_qa_chain(self):
        """
        Get the initialized QA chain.
        
        Returns:
            The QA chain or None if not initialized
        """
        if not self.initialized or self.qa_chain is None:
            success = self.initialize()
            if not success:
                return None
                
        return self.qa_chain
    
    def status(self) -> Dict[str, Any]:
        """
        Get the status of the LLM manager.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "initialized": self.initialized,
            "model_name": os.path.basename(self.model_path),
            "use_small_model": self.use_small_model,
            "context_window": self.context_window,
            "batch_size": self.batch_size
        }
    
    def cleanup(self) -> None:
        """Release resources used by the LLM."""
        if self.llm is not None:
            try:
                # Clean up LLM resources
                del self.llm
                self.llm = None
                self.qa_chain = None
                self.initialized = False
                logger.info("LLM resources released")
            except Exception as e:
                logger.error(f"Error cleaning up LLM: {e}")


# Create a singleton instance
llm_manager = LLMManager()

# For testing purposes
if __name__ == "__main__":
    # Initialize the LLM
    success = llm_manager.initialize()
    
    if success:
        # Test the LLM
        llm = llm_manager.get_llm()
        response = llm("What is a language model?")
        print(f"LLM response: {response}")
    else:
        print("Failed to initialize LLM") 