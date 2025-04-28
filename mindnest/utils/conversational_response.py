"""
Conversational response generator for MindNest's natural language interactions.
This module handles the generation of human-like responses for conversational queries.
"""

from typing import Optional, Dict, Any, List
import time
import random

class ConversationalResponseGenerator:
    """
    A class that handles generating conversational responses using a language model.
    Provides methods for maintaining context, updating query history, and generating
    coherent responses in a conversation.
    """
    
    def __init__(self, llm=None, context_window_size: int = 5):
        """
        Initialize the conversational response generator.
        
        Args:
            llm: The language model to use for generating responses
            context_window_size: Number of previous interactions to maintain in context
        """
        self.llm = llm
        self.context_window_size = context_window_size
        self.conversation_history = []
        self.last_query_time = 0
        self.session_start_time = time.time()
        
        # Standard responses for when LLM is not available
        self.standard_responses = [
            "I'm here to help with your questions about documents and information.",
            "How can I assist you with finding information today?",
            "I can help answer your questions based on the documents I have access to.",
            "Feel free to ask me anything about the information stored in the system.",
            "I'm designed to help you find and understand information in your documents."
        ]
    
    def generate_response(self, query: str, model_capabilities: Dict[str, Any] = None) -> str:
        """
        Generate a response to the given query using the language model.
        
        Args:
            query: The user's query
            model_capabilities: Dictionary containing the capabilities of the model
            
        Returns:
            A string response to the query
        """
        # Update conversation history and timing
        current_time = time.time()
        time_since_last_query = current_time - self.last_query_time
        self.last_query_time = current_time
        
        # Check if this is a new conversation (more than 5 minutes since last query)
        if time_since_last_query > 300:  # 5 minutes in seconds
            self.conversation_history = []
        
        # Try to use the LLM if available
        if self.llm is not None:
            try:
                # Add the current query to history
                self.conversation_history.append({"role": "user", "content": query})
                
                # Maintain history size
                if len(self.conversation_history) > self.context_window_size * 2:
                    self.conversation_history = self.conversation_history[-self.context_window_size * 2:]
                
                # Format the conversation history for the LLM
                formatted_history = self._format_conversation_history()
                
                # Create a system message based on model capabilities
                system_message = self._create_system_message(model_capabilities)
                
                # Generate response
                response = self._generate_llm_response(system_message, formatted_history)
                
                # Add the response to history
                self.conversation_history.append({"role": "assistant", "content": response})
                
                return response
            except Exception as e:
                print(f"Error generating conversational response: {e}")
                # Fall back to standard responses if there's an error
        
        # Fallback to standard responses if LLM is not available
        return self._get_standard_response(query)
    
    def _format_conversation_history(self) -> str:
        """Format the conversation history for the language model."""
        formatted = ""
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            formatted += f"{role.capitalize()}: {content}\n"
        
        # Add the assistant prefix for the next response
        formatted += "Assistant: "
        return formatted
    
    def _create_system_message(self, model_capabilities: Dict[str, Any]) -> str:
        """Create a system message based on model capabilities."""
        system_message = "You are a helpful AI assistant for MindNest. "
        
        if model_capabilities:
            if model_capabilities.get("supports_memory", False):
                system_message += "You can remember previous parts of this conversation. "
            
            if model_capabilities.get("supports_reasoning", False):
                system_message += "Provide thoughtful and reasoned responses. "
                
            max_tokens = model_capabilities.get("max_output_tokens", 1024)
            if max_tokens < 500:
                system_message += "Keep your responses concise and to the point. "
        else:
            system_message += "Keep your responses concise, helpful and to the point. "
            
        system_message += "Respond in a natural, conversational manner."
        return system_message
    
    def _generate_llm_response(self, system_message: str, conversation_history: str) -> str:
        """Generate a response using the language model."""
        if not self.llm:
            return self._get_standard_response("")
            
        try:
            # Create the prompt with system message and conversation history
            prompt = f"{system_message}\n\n{conversation_history}"
            
            # Generate response
            print(f"Invoking LLM with prompt: {prompt[:50]}...")
            response = self.llm.invoke(prompt)
            
            # Clean up response if needed
            response = response.strip()
            
            return response
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._get_standard_response("")
    
    def _get_standard_response(self, query: str) -> str:
        """Get a standard response when the LLM is not available."""
        # For greeting queries, respond with a greeting
        greeting_terms = ["hello", "hi", "hey", "greetings"]
        if any(term in query.lower() for term in greeting_terms):
            return "Hello! How can I assist you today with your questions?"
            
        # For thank you queries, respond with acknowledgment
        thank_terms = ["thank", "thanks"]
        if any(term in query.lower() for term in thank_terms):
            return "You're welcome! Is there anything else I can help you with?"
            
        # Otherwise, return a random standard response
        return random.choice(self.standard_responses)
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update the conversation context with new information.
        
        Args:
            key: The context key to update
            value: The value to store
        """
        # This would store contextual information about the conversation
        # that could be used for more sophisticated response generation
        pass
    
    def handle_feedback(self, query: str, response: str, feedback: str) -> None:
        """
        Process feedback about a previous response.
        
        Args:
            query: The original query
            response: The response that received feedback
            feedback: The feedback (e.g., "positive", "negative")
        """
        # This would handle user feedback to improve future responses
        # Could be used for continuous learning
        pass


def create_conversational_response_generator(llm=None):
    """
    Factory function to create and return a conversational response generator.
    
    Args:
        llm: The language model to use
        
    Returns:
        A ConversationalResponseGenerator instance
    """
    return ConversationalResponseGenerator(llm=llm) 