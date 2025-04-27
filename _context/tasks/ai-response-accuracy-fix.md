# AI RESPONSE ACCURACY FIX FOR LIGHTWEIGHT AND BIG MODELS

## SUMMARY
Improve the accuracy and appropriateness of AI responses in the MindNest application for both lightweight (small) and big language models by optimizing prompts, context handling, and response generation based on model capabilities.

## REQUIREMENTS
- Fix AI response accuracy issues for both model sizes
- Optimize context handling based on model capabilities
- Improve query categorization
- Enhance prompt engineering for different model sizes
- Improve response formatting
- Provide model-specific template customization

## FILE TREE:
- main.py - Main application with `/ask` endpoint
- utils/responses.py - Contains conversation response templates 
- utils/query_optimization.py - Handles query categorization and optimization
- utils/llm_manager.py - Manages LLM initialization and configuration
- utils/query_classification/classifier.py - AI-based query classifier

## IMPLEMENTATION DETAILS

### Current Implementation Analysis
The MindNest application has several components for handling AI responses:

1. **Query Classification System**
   - Uses regex patterns for lightweight classification
   - Provides AI-based embedding classification
   - Categorizes queries as: DOCUMENT_SEARCH, DOCUMENT_QUERY, CONCISE_QUERY, or CONVERSATION

2. **Response Generation**
   - `/ask` endpoint in main.py handles all query processing
   - Different response paths based on query type
   - Pre-written templates for CONVERSATION responses
   - LLM-generated responses for DOCUMENT_QUERY and CONCISE_QUERY

3. **Model Management**
   - Supports two model sizes: large (13B) and small (7B)
   - Same prompt templates used for both model sizes
   - Different context window sizes (4096 vs 2048)

### Issues to Address

1. **Context Handling Issues**
   - Same document retrieval count for both models
   - No adaptation to model context window differences
   - Inefficient use of context window

2. **Query Categorization Issues**
   - Classification may not be optimal for all query types
   - Thresholds are static and not model-specific

3. **Prompt Engineering Issues**
   - Generic prompts not optimized for model size
   - No model-specific instruction tuning

4. **Response Formatting Issues**
   - No explicit formatting guidance based on model capabilities
   - No handling for model hallucinations or quality differences

5. **Template Customization Issues**
   - No model-specific templates
   - No handling for different model strengths/limitations

### Solution Approach

1. **Model-Aware Context Handling**
   - Adjust document retrieval count based on model size
   - Optimize chunk size based on model context window
   - Implement dynamic context truncation based on model capabilities

2. **Enhanced Query Categorization**
   - Refine classification heuristics
   - Add model-specific threshold adjustments
   - Improve conversation detection

3. **Model-Specific Prompt Engineering**
   - Create separate prompt templates for small vs large models
   - Adjust instruction complexity based on model capabilities
   - Add explicit formatting directives tailored to each model

4. **Response Structure Optimization**
   - Add quality control checks for small model responses
   - Implement fallback mechanisms for lower-quality responses
   - Provide explicit structure guidance for smaller models

5. **Template and Style Customization**
   - Create model-specific templates based on capabilities
   - Adjust verbosity and detail level based on model size
   - Implement confidence indicators for responses

### Code-Level Implementation Details

#### Task 1: Update the LLM manager to provide model capabilities information

```python
# In utils/llm_manager.py

class LLMManager:
    # ... existing code ...
    
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
            
        return capabilities
```

#### Task 2: Modify document retrieval in `/ask` endpoint based on model size

```python
# In main.py

@app.post("/ask")
async def ask(request: Request):
    # ... existing code ...
    
    # Get model capabilities for context-aware processing
    model_capabilities = llm_manager.get_capabilities()
    
    # ... existing code for query classification ...
    
    if query_type == "CONCISE_QUERY":
        # Use model-specific document limit for concise queries
        k = model_capabilities["concise_limit"]
        docs = vectorstore.similarity_search(query, k=k)
        # ... rest of the handling ...
    
    elif query_type == "DOCUMENT_QUERY":
        # Use model-specific document limit for regular queries
        k = model_capabilities["document_limit"]
        docs = vectorstore.similarity_search(query, k=k)
        # ... rest of the handling ...
```

#### Task 3: Create separate prompt templates for different model sizes

```python
# In main.py

# Define model-specific prompts
def get_model_specific_prompts(model_capabilities):
    """Get prompt templates optimized for the current model size."""
    model_size = model_capabilities["model_size"]
    
    if model_size == "small":
        # Simpler, more structured prompts for small models
        concise_template = """
        Answer the question below using ONLY the provided context information.
        Keep your answer in 1-2 short sentences.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        SHORT ANSWER:
        """
        
        document_template = """
        Answer the following question using ONLY information from the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        ANSWER (be clear and direct):
        """
    else:
        # More flexible prompts for larger models
        concise_template = """
        Answer the following question in a single concise paragraph of at most 2-3 sentences.
        Be direct, clear, and brief without unnecessary details.
        Use only the information provided in the context.
        
        Context information:
        {context}
        
        Question: {query}
        
        Concise Answer (2-3 sentences maximum):
        """
        
        document_template = """
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
    
    return {
        "concise": PromptTemplate.from_template(concise_template),
        "document": PromptTemplate.from_template(document_template)
    }

# Then in the ask endpoint:
@app.post("/ask")
async def ask(request: Request):
    # ... existing code ...
    
    # Get model capabilities and optimized prompts
    model_capabilities = llm_manager.get_capabilities()
    model_prompts = get_model_specific_prompts(model_capabilities)
    
    if query_type == "CONCISE_QUERY":
        # ... existing retrieval code ...
        
        # Use model-specific prompt
        concise_chain = load_qa_chain(llm, chain_type="stuff", 
                                      prompt=model_prompts["concise"])
        # ... rest of handling ...
```

#### Task 4: Optimize query classification thresholds based on model size

```python
# In utils/query_optimization.py

@lru_cache(maxsize=128)
def categorize_query(query, model_size="large"):
    """
    Categorize a query with model-specific thresholds.
    
    Args:
        query (str): The user's query text
        model_size (str): "small" or "large" to adjust thresholds
        
    Returns:
        tuple: (query_type, processed_query)
    """
    query = query.strip()
    
    # Check if this is a search query (same for both models)
    if query.lower().startswith("find "):
        return "DOCUMENT_SEARCH", query[5:]
    
    # Model-specific adjustments for query categorization
    # For small models, be more aggressive about categorizing as CONCISE_QUERY
    # to avoid overwhelming the model
    if model_size == "small":
        # Additional concise patterns for small models
        simple_query_patterns = [
            r"^.{0,50}$",  # Very short queries
            r"^(what|how|why|when|where|which|who)\s+.{0,30}$",  # Simple question patterns
        ]
        
        for pattern in simple_query_patterns:
            if re.search(pattern, query.lower()):
                return "CONCISE_QUERY", query
    
    # Standard concise patterns (for both models)
    # ... existing patterns ...
    
    # Conversation detection with model-specific thresholds
    conversation_likelihood = is_likely_conversation(query)
    if model_size == "small" and conversation_likelihood > 0.4:
        # Lower threshold for small models to avoid complex queries
        return "CONVERSATION", query
    elif model_size == "large" and conversation_likelihood > 0.7:
        # Higher threshold for larger models that can handle more
        return "CONVERSATION", query
    
    # Use is_likely_document_query for further classification
    if is_likely_document_query(query):
        # For small models, prefer concise queries for shorter questions
        if model_size == "small" and len(query.split()) < 8:
            return "CONCISE_QUERY", query
        return "DOCUMENT_QUERY", query
    
    # Otherwise, it's likely just conversational
    return "CONVERSATION", query
```

#### Task 5: Implement model-specific response formatting directives

This will be incorporated into the prompt templates, but we'll also add a post-processing function:

```python
# In main.py

def format_response(response_text, model_capabilities):
    """Apply model-specific formatting to ensure quality responses."""
    model_size = model_capabilities["model_size"]
    
    # For small models, apply more aggressive formatting and checks
    if model_size == "small":
        # Remove potential hallucinations indicated by uncertain language
        response_text = re.sub(r'(?i)I\'m not sure|I don\'t know|I believe|probably|might be|possibly',
                              '', response_text)
        
        # Ensure the response isn't too long for small models
        words = response_text.split()
        if len(words) > 100:
            response_text = ' '.join(words[:100]) + '...'
    
    # For all models, clean up formatting
    response_text = response_text.strip()
    
    # Remove any extra newlines (more than 2 consecutive)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    return response_text

# Use in the ask endpoint for both query types:
@app.post("/ask")
async def ask(request: Request):
    # ... existing code ...
    
    if query_type == "CONCISE_QUERY" or query_type == "DOCUMENT_QUERY":
        # ... existing code to get answer ...
        
        # Apply model-specific formatting
        formatted_response = format_response(answer["output_text"], model_capabilities)
        
        return {"text": formatted_response, "sources": [doc.metadata["source"] for doc in docs]}
```

#### Task 6: Add quality control mechanisms for small model responses

```python
# In main.py

def validate_response_quality(response, query, model_capabilities):
    """
    Validate the quality of the model's response.
    Returns the original response if good, or a fallback if poor quality.
    """
    model_size = model_capabilities["model_size"]
    
    # Only apply strict validation to small models
    if model_size != "small":
        return response
    
    # Check for potential low-quality indicators
    low_quality = False
    
    # 1. Too short responses
    if len(response.split()) < 5:
        low_quality = True
    
    # 2. Repetitive content
    words = response.lower().split()
    unique_words = set(words)
    if len(words) > 0 and len(unique_words) / len(words) < 0.5:
        low_quality = True
    
    # 3. Nonsensical or incomplete sentences
    if response.count('.') == 0 or not response.strip().endswith(('.', '?', '!')):
        low_quality = True
    
    # 4. Response doesn't seem to address the query
    query_keywords = set([w.lower() for w in query.split() if len(w) > 3])
    response_text = response.lower()
    matches = sum(1 for word in query_keywords if word in response_text)
    if len(query_keywords) > 0 and matches / len(query_keywords) < 0.2:
        low_quality = True
    
    # If low quality detected, provide a fallback response
    if low_quality:
        return generate_fallback_response(query)
    
    return response

def generate_fallback_response(query):
    """Generate a fallback response when the model output is low quality."""
    fallback = (
        "Based on the documentation, I can't provide a complete answer to this question. "
        "The available information is limited, but you might find relevant details by "
        "searching for specific keywords related to your question."
    )
    return fallback
```

#### Task 7: Update conversation handling to detect and redirect difficult queries

```python
# In utils/responses.py

def assess_query_complexity(query):
    """
    Assess the complexity of a query on a scale of 1-5.
    Higher numbers indicate more complex queries.
    """
    complexity = 1  # Start with baseline complexity
    
    # Longer queries tend to be more complex
    words = query.split()
    if len(words) > 15:
        complexity += 1
    if len(words) > 25:
        complexity += 1
    
    # Queries with multiple questions are more complex
    question_marks = query.count('?')
    if question_marks > 1:
        complexity += 1
    
    # Queries with advanced terms suggest complexity
    advanced_terms = ['compare', 'contrast', 'analyze', 'synthesize', 'evaluate', 
                      'implications', 'consequences', 'relationship', 'causation',
                      'methodology', 'framework', 'implementation', 'architecture']
    
    if any(term in query.lower() for term in advanced_terms):
        complexity += 1
    
    # Cap at 5
    return min(complexity, 5)

def get_conversation_response(mode="professional", query_complexity=1):
    """
    Get a response based on conversation mode and query complexity.
    
    Args:
        mode (str): Conversation mode
        query_complexity (int): Complexity level 1-5
    """
    # For highly complex queries (4-5), provide a specialized response
    if query_complexity >= 4:
        complex_responses = [
            "That's a complex question that would be better addressed with specific document references. Try rephrasing as a document query.",
            "For detailed technical questions like this, I'd need to reference specific documentation. Could you ask about a specific document?",
            "This seems like a complex topic that would benefit from searching the documentation. Try asking a more specific document-focused question.",
        ]
        return random.choice(complex_responses)
    
    # For medium complexity (2-3), provide a modified standard response
    elif query_complexity >= 2:
        # Get standard response but modify it to acknowledge complexity
        standard = get_standard_response(mode)
        complexity_additions = [
            " Your question seems somewhat detailed. ",
            " That's a good question that might require document context. ",
            " For more detailed information on this topic, "
        ]
        return random.choice(complexity_additions) + standard
    
    # For simple queries (1), use the existing responses
    else:
        return get_standard_response(mode)
        
def get_standard_response(mode):
    """Get a standard response based on the mode."""
    if mode.lower() == "passive_aggressive":
        return random.choice(PASSIVE_AGGRESSIVE_RESPONSES)
    elif mode.lower() == "humorous":
        return random.choice(HUMOROUS_RESPONSES)
    else:  # Default to professional
        return random.choice(PROFESSIONAL_RESPONSES)
```

#### Task 8: Implement context window optimization for both models

```python
# In main.py

def optimize_context_for_model(docs, query, model_capabilities):
    """
    Optimize document context based on model capabilities.
    
    Args:
        docs: Retrieved documents
        query: User query
        model_capabilities: Model capabilities dict
        
    Returns:
        Optimized document list
    """
    model_size = model_capabilities["model_size"]
    context_window = model_capabilities["context_window"]
    
    # For small models, be more aggressive with context reduction
    if model_size == "small":
        # Calculate approximate max tokens per document
        # Assume we need about 25% of context window for query and response
        max_tokens = int((context_window * 0.75) / max(1, len(docs)))
        
        # Truncate document content to fit
        for i, doc in enumerate(docs):
            content = doc.page_content
            # Rough estimate: 1 token ≈ 4 characters
            if len(content) > max_tokens * 4:
                # Prioritize beginning of documents
                docs[i].page_content = content[:max_tokens * 4] + "..."
    else:
        # For larger models, we can be more generous but still optimize
        max_tokens = int((context_window * 0.85) / max(1, len(docs)))
        
        # More balanced truncation for larger models
        for i, doc in enumerate(docs):
            content = doc.page_content
            if len(content) > max_tokens * 4:
                # Keep both beginning and end as these may have important info
                half_length = int(max_tokens * 2)
                docs[i].page_content = content[:half_length] + "..." + content[-half_length:]
    
    return docs
```

#### Task 9: Add fallback mechanisms for potentially low-quality responses

```python
# In main.py

def create_hybrid_response(llm_response, query, docs, model_capabilities):
    """
    Create a hybrid response combining template text with LLM output.
    Used when LLM response might be low quality.
    """
    model_size = model_capabilities["model_size"]
    
    # Only apply for small models
    if model_size != "small":
        return llm_response
    
    # Check confidence in the response
    confidence = assess_response_confidence(llm_response, query)
    
    if confidence < 0.4:  # Low confidence threshold
        # Create a templated response with the LLM output incorporated
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        source_str = ", ".join(sources[:2])
        if len(sources) > 2:
            source_str += f" and {len(sources)-2} more"
            
        # Hybrid response template
        hybrid = (
            f"Based on information from {source_str}, I found: "
            f"{llm_response}\n\n"
            f"Note: This information is summarized from the documentation and may be incomplete."
        )
        return hybrid
    
    return llm_response

def assess_response_confidence(response, query):
    """
    Assess confidence level in the model's response.
    Returns a score between 0-1 (higher is better).
    """
    # Simple heuristics for confidence assessment
    confidence = 0.5  # Start with neutral confidence
    
    # Length-based assessment (too short or too long are suspicious)
    words = response.split()
    if len(words) < 10:
        confidence -= 0.2
    elif len(words) > 200:
        confidence -= 0.1
    
    # Check for hedging language
    hedging_terms = ['i think', 'probably', 'might', 'may be', 'possibly', 
                     'i believe', 'perhaps', 'seems', 'could be']
    hedges = sum(1 for term in hedging_terms if term in response.lower())
    confidence -= 0.1 * min(3, hedges)  # Max penalty of 0.3
    
    # Check for query term presence
    query_terms = [term.lower() for term in query.split() if len(term) > 3]
    matched = sum(1 for term in query_terms if term in response.lower())
    if query_terms:
        term_match_ratio = matched / len(query_terms)
        confidence += 0.2 * term_match_ratio  # Max bonus of 0.2
    
    # Ensure confidence is within 0-1 range
    return max(0.0, min(1.0, confidence))
```

#### Task 10: Update the QA chain prompt templates for model-specific guidance

```python
# In main.py - initialize_qa_chain() function

def initialize_qa_chain():
    """Initialize the question-answering chain with the LLM."""
    global qa_chain, llm, max_context_tokens, use_small_model
    
    try:
        if llm is None:
            print("LLM not initialized. QA chain initialization skipped.")
            return False
    
        print("Initializing QA chain...")
        
        from langchain.chains.question_answering import load_qa_chain
        from langchain.prompts import PromptTemplate
        
        # Model-specific prompt templates
        if use_small_model:
            # Structured, simpler template for small models
            template = """
            Answer the question based ONLY on the context provided below.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {query}
            
            INSTRUCTIONS:
            - Use ONLY information from the context
            - Keep your answer clear and direct
            - If the question asks for a brief answer, be very concise
            - Format your answer in simple paragraphs
            - If you don't know, say "The documentation doesn't provide this information"
            
            ANSWER:
            """
        else:
            # More flexible template for larger models
            template = """
            Answer the following question based on the provided context. 
            
            If the question asks for a brief or concise answer, keep your response short and to the point.
            If the question asks for a summary or a definition in one sentence, provide exactly that.
            Focus on answering the exact question without adding irrelevant information.
            Only include information that is directly relevant to answering the specific question.
            If the information isn't in the context, acknowledge that the documentation doesn't cover it.
            
            Context pieces:
            {context}
            
            Question: {query}
            
            Answer:
            """
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Create a QA chain with model-specific parameters
        chain_type = "stuff"  # Standard for both models
        
        # Create the chain
        qa_chain = load_qa_chain(llm, chain_type=chain_type, prompt=QA_CHAIN_PROMPT)
        
        print("QA chain initialized successfully with model-specific template")
        return True
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        import traceback
        traceback.print_exc()
        return False
```

## TODO LIST
[ ] 1. Update the LLM manager to provide model capabilities information
    - Modify utils/llm_manager.py to track model type (small/large)
    - Add get_capabilities() method to return model-specific settings
    - Include context window size, token capacity, and other relevant metrics

[ ] 2. Modify document retrieval in `/ask` endpoint based on model size
    - Update main.py to get model capabilities from the LLM manager
    - Adjust document count (k parameter) in similarity_search calls
    - For small models: k=2 for CONCISE_QUERY, k=3 for DOCUMENT_QUERY
    - For large models: k=3 for CONCISE_QUERY, k=4-5 for DOCUMENT_QUERY

[ ] 3. Create separate prompt templates for different model sizes
    - Add new model-specific prompt templates in main.py
    - For small models: create simpler, more structured templates
    - For large models: allow more flexibility and complexity
    - Update PromptTemplate instantiation to use the correct template based on model size

[ ] 4. Optimize query classification thresholds based on model size
    - Update utils/query_optimization.py to consider model capabilities
    - Add model_type parameter to categorize_query() function
    - Adjust pattern matching thresholds based on model capabilities

[ ] 5. Implement model-specific response formatting directives
    - Update prompts to include explicit formatting instructions
    - For small models: add clear section markers and structure
    - For large models: maintain current flexibility with improved guidance

[ ] 6. Add quality control mechanisms for small model responses
    - Create post-processing function to validate smaller model outputs
    - Add validation for answer length, relevance, and formatting
    - Implement fallback to templated responses for poor quality answers

[ ] 7. Update conversation handling to detect and redirect difficult queries
    - Enhance utils/responses.py to handle different query complexities
    - Add complexity assessment for conversation queries
    - Provide more specific conversation responses for complex queries

[ ] 8. Implement context window optimization for both models
    - Add context truncation logic based on model context window
    - Adjust text chunking to fit model limitations
    - Prioritize recent and relevant content in context

[ ] 9. Add fallback mechanisms for potentially low-quality responses
    - Create fallback templates for when LLM responses appear low-quality
    - Implement confidence scoring for responses
    - Create hybrid responses combining template text with LLM output

[ ] 10. Update the QA chain prompt templates for model-specific guidance
    - Modify initialize_qa_chain() to use model-specific prompts
    - Create separate QA chain instances for different model types
    - Update chain initialization to respect model capabilities 