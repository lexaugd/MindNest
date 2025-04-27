"""
Example queries for different query types.
These are used by the AI-based query classifier.
"""

# Conversation queries are casual interactions not related to documents
CONVERSATION_QUERIES = [
    # Greetings
    "hello",
    "hi there",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    
    # How are you variations
    "how are you",
    "how are you doing",
    "how's it going",
    "how have you been",
    "what's up",
    "how do you do",
    
    # Personal questions
    "what's your name",
    "who are you",
    "tell me about yourself",
    "what can you do",
    "what are your capabilities",
    
    # Thanks/acknowledgments
    "thank you",
    "thanks",
    "appreciate it",
    "that's helpful",
    "got it",
    "I understand",
    
    # Casual questions
    "what time is it",
    "what's the weather like",
    "how's the weather today",
    "do you like music",
    "what do you think about AI",
    
    # Farewells
    "goodbye",
    "bye",
    "see you later",
    "take care",
    "until next time",
    
    # Chitchat
    "how's your day",
    "do you enjoy your work",
    "are you having a good day",
    "you're smart",
    "you're helpful",
    "that's interesting",
    
    # Small talk
    "it's a nice day today",
    "I'm tired",
    "I'm excited",
    "I'm learning about AI",
    "I'm working on a project"
]

# Document queries ask for specific information from the documents
DOCUMENT_QUERIES = [
    # What is X
    "what is the object wrapper system",
    "what is ListWrapper",
    "what is MapWrapper",
    "what is a wrapper factory",
    "what is the null object pattern",
    
    # How does X work
    "how does ListWrapper work",
    "how does MapWrapper work",
    "how does the object wrapper system work",
    "how does the factory pattern work",
    "how does null safety work",
    
    # Explain X
    "explain the wrapper system",
    "explain the factory pattern",
    "explain null safety",
    "explain chain of responsibility",
    "explain type conversion",
    
    # What are the benefits
    "what are the benefits of wrappers",
    "what advantages do object wrappers provide",
    "what are the features of ListWrapper",
    "what problems does the wrapper system solve",
    
    # How to X
    "how to use ListWrapper",
    "how to implement a custom wrapper",
    "how to handle null values",
    "how to convert between types",
    "how to chain wrappers",
    
    # Tell me about X
    "tell me about the wrapper system",
    "tell me about ListWrapper",
    "tell me about MapWrapper",
    "tell me about type conversion",
    "tell me about null safety",
    
    # Show me X
    "show me examples of wrappers",
    "show me how to use ListWrapper",
    "show me the implementation details",
    "show me null safety in action",
    "show me best practices"
]

# Concise queries explicitly ask for brief, summarized information
CONCISE_QUERIES = [
    # Explicit brevity keywords
    "summarize the wrapper system",
    "summarize ListWrapper",
    "briefly explain MapWrapper",
    "concisely describe object wrappers",
    "short description of factory pattern",
    
    # Sentence limit requests
    "explain ListWrapper in one sentence",
    "what is MapWrapper in a single sentence",
    "describe object wrappers in one paragraph",
    "what is null safety in a few words",
    "tell me about type conversion in one line",
    
    # "Brief" keyword
    "brief explanation of wrappers",
    "brief overview of ListWrapper",
    "brief summary of null safety",
    "brief description of factory pattern",
    
    # "Quick" keyword
    "quick explanation of wrappers",
    "quick overview of MapWrapper",
    "quick summary of null safety",
    
    # "TLDR" style
    "tldr wrapper system",
    "tldr ListWrapper",
    "tldr factory pattern"
]

# Searches are explicitly looking for document content
DOCUMENT_SEARCH_QUERIES = [
    # Find X 
    "find wrapper examples",
    "find ListWrapper documentation",
    "find null safety patterns",
    "find type conversion methods",
    "find factory pattern examples",
    
    # Search for X
    "search for wrapper system",
    "search for ListWrapper",
    "search for MapWrapper",
    "search for null safety",
    "search for type conversion",
]

# All query types combined
ALL_EXAMPLE_QUERIES = {
    "CONVERSATION": CONVERSATION_QUERIES,
    "DOCUMENT_QUERY": DOCUMENT_QUERIES,
    "CONCISE_QUERY": CONCISE_QUERIES,
    "DOCUMENT_SEARCH": DOCUMENT_SEARCH_QUERIES
} 