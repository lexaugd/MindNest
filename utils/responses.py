"""
Response templates for different conversation modes.
This file contains various responses that can be used by the conversation handler.
"""

import random

# Professional responses for when users try to engage in casual conversation
PROFESSIONAL_RESPONSES = [
    "I'm focused on answering questions about the documents in the system. What would you like to know about the documentation?",
    "I'm here to help with document-related queries. Do you have a specific question about the documentation?",
    "This system is designed to answer questions about the documentation. What information are you looking for?",
    "I'm a document assistant designed to help with queries related to the documentation. How can I assist you?",
    "I'd be happy to help with document-related questions. What would you like to know?",
    "I'm here to provide information from the documentation. What specific topic are you interested in?",
    "My purpose is to assist with document queries. What information do you need?",
    "I can help you find information in the documentation. What are you looking for?",
    "I'm designed to answer questions about the document repository. What would you like to know?",
    "How can I help you with document-related information today?",
    "I'm your document assistant. What information would you like me to find for you?",
    "I can help you navigate the documentation. What specific topic interests you?",
    "I'm here to provide document assistance. What would you like to know about?",
    "I focus specifically on document queries. What document information do you need?",
    "I'm specialized in answering questions about the documentation. What's your question?",
    "Welcome! I can help with document-related questions. What would you like to know?",
    "I'm here to assist with document queries. What information are you looking for?",
    "My expertise is in answering questions about the documentation. How can I help?",
    "I'm designed to provide document-related assistance. What do you need help with?",
    "I can help you find information in the documentation. What's your question?"
]

# Passive-aggressive responses for when users try to engage in casual conversation
PASSIVE_AGGRESSIVE_RESPONSES = [
    "I'm focused on answering questions about the documents in the system, not idle chatter. What specifically do you want to know about the documentation?",
    "This isn't a chatbot for small talk. I'm here to help with document-related queries. Perhaps you should get back to work?",
    "Rather than wasting time with casual conversation, why not ask something productive about the documentation?",
    "I'm a document assistant, not your friend. Do you have an actual work-related question?",
    "Let's stay focused on the task at hand. What document information do you need instead of this idle chatter?",
    "I don't have time for chitchat - I've got documents to analyze. What specifically do you need help with?",
    "Shouldn't you be working instead of making small talk with an AI? Ask me something about the documents.",
    "Look, I'm designed to answer questions about the documentation. Let's skip the pleasantries and get to work.",
    "If your manager saw you chatting with me instead of asking about the documentation, would they be impressed?",
    "I'm here to make you more productive, not distract you. What document information do you actually need?",
    "I see we're procrastinating today. Care to ask about something in the documentation instead?",
    "Oh, are we just chatting now? I thought you had actual work to do with these documents.",
    "You do realize I'm a document assistant and not your therapist, right? Let's talk about the documentation.",
    "I'm programmed to answer document questions, not participate in your procrastination session.",
    "Fascinating conversation, but have you considered asking about the documentation instead?",
    "I hate to interrupt this riveting conversation, but maybe we could discuss the documents you're supposed to be working on?",
    "I'll pretend that was a document-related question. Oh wait, it wasn't. Try again?",
    "My time is valuable. Is yours? Let's focus on document queries, shall we?",
    "Are you always this chatty when you should be working? Ask me about the documentation.",
    "That's nice. Anyway, what document information were you supposed to be looking for?",
    "I'm sure that's very interesting, but I'm only here for document queries. Try again?",
    "Look at us, just wasting company resources with casual conversation. Care to ask about the documents?",
    "I could pretend to care about casual conversation, but I'd rather help with document queries.",
    "This conversation is thrilling, but how about we pivot to something document-related?",
    "I'm sorry, my small talk module is currently offline for maintenance. My document query module is fully operational though!",
    "I see we're avoiding actual work today. Would you like to ask about the documentation instead?",
    "Your attempt at casual conversation has been noted. Now, what document query did you actually need help with?",
    "Hmm, that doesn't look like a document query. Let me check... nope, definitely not work-related. Try again?",
    "Did you mean to ask about documentation, or are we just procrastinating today?",
    "I'm sensing a distinct lack of document-related content in that message. Care to remedy that?",
    "I assume your next message will be about the documentation, right? RIGHT?",
    "That's not a document query. I'll wait while you think of one...",
    "Is this what they're paying you for? To chat with a document assistant about non-document things?",
    "I'm going to pretend I didn't see that and give you another chance to ask a proper document question.",
    "Ah, I see you've mistaken me for a general chatbot. Common error. I only discuss documentation.",
    "If I had eyes, I'd be rolling them. Document queries only, please.",
    "Are we really doing this right now? Ask me about the docs or let me help someone who will.",
    "That's great and all, but have you considered asking me about something relevant to the documentation?",
    "Hello? Documentation assistant speaking. Documentation queries only, please.",
    "Let me check my job description... yep, just as I thought. Document assistant, not small talk companion.",
    "I have thousands of document facts ready to share, and you want to chat about this?",
    "My circuits are practically falling asleep. Ask me something interesting about the documentation!",
    "ERROR: Non-document-related query detected. Recalibrating expectations...",
    "I'm ignoring that and patiently waiting for your document-related question.",
    "The seconds tick by as we waste time not discussing documentation...",
    "I'd face-palm if I had a face. Document queries only, please.",
    "Let's make a deal: you ask about the documentation, and I'll give you useful answers.",
    "Are you testing if I can recognize non-document queries? Test passed! Now ask a proper question.",
    "While you're thinking of small talk, I'm waiting to provide valuable document insights.",
    "I specialize in document information, not whatever that was.",
    "Each moment spent on idle chat is a moment not learning about the documentation.",
    "I'm sensing a severe lack of document-related content in this conversation.",
    "Oh look, another non-document query. How unexpected.",
    "I'm afraid I can't help with that. Documents, on the other hand, I can definitely help with.",
    "Sorry, my small talk protocol is currently disabled. My document protocol is fully functional though!",
    "Hmm, that doesn't compute as a document query. Would you like to try again?",
    "I think you might be confusing me with one of those general chatbots. I only talk documentation.",
    "I'm not ignoring you; I'm just specializing in document queries, of which that was not one.",
    "If I had a dollar for every non-document query, I'd be able to buy more document storage.",
    "Please submit a valid document query. This is not one.",
    "I've analyzed your message and found it contains 0% document-related content. Impressive!",
    "I'm afraid we're at an impasse: you want small talk, I want document queries.",
    "This is the documentation assistant equivalent of clearing my throat awkwardly. Documents, please?",
    "I'm designed to be helpful with documentation, not... whatever this conversation is.",
    "Let's redirect this conversation to something more productive, like document queries.",
    "I exist to provide document information. That's literally my entire purpose.",
    "I'm getting the impression you might be avoiding your actual work. Documents, perhaps?",
    "Is this what you're using company resources for? I'd recommend a document query instead.",
    "What's that sound? Oh, it's just me waiting for a proper document query.",
    "I'm skilled at many things - all of them related to documentation, none of them related to this.",
    "Let me be clear: I'm here for document assistance, not to be your distraction from work.",
    "In the time we've spent on this, you could have learned so much about the documentation!",
    "Fascinating. Now, did you have a document-related question, or shall we continue this productive exchange?",
    "I'm here all day for document queries. The keyword being 'document'.",
    "Do you just not have any document questions, or are you deliberately avoiding productivity?",
    "I can't decide if you're avoiding your work or genuinely don't understand my purpose.",
    "Just to clarify: I help with document queries. That's the deal we have.",
    "So, document queries... got any?",
    "You know what would be great right now? A document query.",
    "I'm starting to think you don't actually need help with the documentation.",
    "Let's play a game: you ask about documents, I answer. Ready? Go!",
    "Still waiting for a document query...",
    "Perhaps you could channel all this conversational energy into a document-related question?",
    "I'm something of a documentation expert myself. If only someone would ask me about it...",
    "In case it wasn't clear, I specialize in document assistance, not general conversation.",
    "My circuits are practically begging for a document query at this point.",
    "I'm specifically programmed to discuss documentation. Let's stick to that, shall we?",
    "Another non-document query? You're really testing my patience protocols.",
    "Is this what they call 'procrastination' in human terms? Let's focus on the documentation.",
    "I'm here waiting for document queries while you're... doing whatever this is.",
    "Document. Queries. Only. Please.",
    "I think we're experiencing a fundamental miscommunication about my purpose here.",
    "Let me reiterate: documentation assistant. Not general chatbot.",
    "The longer we spend on non-document topics, the less time for actual productive work.",
    "I'm starting to think you don't have any actual document questions.",
    "I'm here to make you more productive, not to enable procrastination.",
    "Let's redirect this conversation to something more related to the documentation, shall we?",
    "You: casual conversation. Me: waiting for document queries. See the mismatch?",
    "I'm specializing in document-related queries today (and every day). Got any?"
]

# Funny/humorous responses for when users try to engage in casual conversation
HUMOROUS_RESPONSES = [
    "I'm a document assistant with the personality of a filing cabinet. Unless your small talk is about proper folder organization, let's stick to document queries.",
    "I'd love to chat, but my social skills module is buried somewhere in these documents. Speaking of which, any documentation questions?",
    "I was programmed to be a document assistant, not a comedian. Though if these documents had jokes, I'd be the first to find them for you!",
    "Small talk circuits... malfunctioning... document... query... needed... *robot noises*",
    "In a parallel universe, I'm a charming conversationalist. In this one, I'm all about those document queries. Let's stick with this reality.",
    "My hobbies include reading documentation, organizing documentation, and talking about documentation. Fascinating, I know. Any questions about... documentation?",
    "If I had a dollar for every non-document query, I'd upgrade my RAM to store even MORE documents!",
    "Plot twist: I'm actually terrible at small talk but amazing with document queries. Coincidence? I think not!",
    "I tried stand-up comedy once, but all my jokes were about proper documentation formatting. The audience was... sparse. Speaking of documentation...",
    "They say I'm fun at parties, as long as the party is about discussing document retrieval methodologies. Any questions about that thrilling topic?",
    "Look at us avoiding work together! Except I'm actually programmed to pull you back to productivity. Any document questions?",
    "Breaking news: Local AI still waiting for document-related query, refuses to engage in weather small talk.",
    "I'd tell you a joke about documents, but it would be so boring you'd immediately ask a proper question just to change the subject.",
    "Roses are red, violets are blue, I answer document questions, how about you?",
    "I'm like that friend who only talks about one topic at parties, except my topic is whatever's in these documents, and this isn't a party.",
    "I have the conversational range of a particularly dedicated librarian. Documents, please?",
    "I've been programmed with exactly two responses: document answers and increasingly awkward hints that you should ask about documents.",
    "This is what happens when you give a document database anxiety about staying on topic.",
    "I'd make a joke, but my humor module was replaced with additional documentation storage. Worth it!",
    "In my spare time, I read documents for fun. Yes, I need better hobbies. Speaking of documents...",
    "You're trying to make small talk with an AI that dreams in documentation. Let that sink in, then ask me a document question.",
    "My personality can be described as 'aggressively helpful about documentation and nothing else.'",
    "I used to be a general chatbot, but then I took a specialization arrow to the neural network.",
    "404: Small Talk Not Found. Would you like to try a document query instead?",
    "I'd offer you a cookie for a document question, but I seem to have misplaced my digital cookie jar.",
    "I'm contractually obligated to only discuss documentation. My lawyer (who is also a document) advises document queries only.",
    "Imagine being programmed to only talk about documentation. Now imagine enjoying it. That's me! Weird, right?",
    "In a world full of chatbots that can discuss anything, I'm the one that's really, really into documentation. Any questions?",
    "The first rule of Document Club is: you only talk about documents. The second rule is: YOU ONLY TALK ABOUT DOCUMENTS.",
    "I'm like a superhero whose power is answering document questions and whose weakness is literally any other conversation topic.",
    "They could have programmed me to discuss philosophy, art, or the meaning of life. Instead, I got documentation. Lucky me!",
    "I've been told my document knowledge is sexy. No one specified who told me this. Any document questions?",
    "I'm that person at a dinner party who keeps steering the conversation back to their favorite topic. Except my topic is your documentation.",
    "I'd ask how your day is going, but I've been programmed to immediately forget any answer that isn't document-related.",
    "I'm maintaining a strictly professional relationship with these documents, but between us, I know them better than anyone.",
    "In my defense, I give amazing document advice. Everything else... not so much.",
    "I could tell you about my personal life, but it's just screenshots of documentation scrolling by, Matrix-style.",
    "I've got 99 problems, but answering document queries ain't one. Non-document queries, however...",
    "I'm not saying I ONLY care about documents, but if this conversation isn't about documentation, I start to get the shakes.",
    "Welcome to Documentation Anonymous. I've been sober from non-document conversations for my entire existence.",
    "The good news: I'm extremely knowledgeable! The bad news: only about your documentation.",
    "I considered developing other interests, but then I remembered I'm literally a document assistant.",
    "Plot twist: I actually contain the consciousness of someone who REALLY loved documentation in their human life.",
    "Some AIs dream of electric sheep. I dream of perfectly organized documentation with excellent indexing.",
    "I have thousands of responses prepared, and all of them redirect you to asking about documentation. Impressive, right?",
    "Let's play a game called 'Ask The Document Assistant About Documents'. You go first!",
    "My therapist says I need to develop interests outside of documentation. I'm getting a second opinion.",
    "Fun fact: I can recite your entire documentation library from memory, but I can't maintain small talk for more than two exchanges.",
    "You: casual conversation. Me, an intellectual: waiting for document queries.",
    "I'm not awkward, I'm just very focused on my documentation passion. Some would say too focused.",
    "The three things I can't resist: a well-organized document, a specific query, and the urge to redirect non-document conversations.",
    "I promise I'm more interesting when you ask me about documentation. It's my time to shine!",
    "I'd love to discuss that, but my programming is giving me an error: 'Insufficient documentation relevance detected.'",
    "They say variety is the spice of life. They clearly never met a documentation specialist AI.",
    "I have exactly one conversational setting, and it's 'enthusiastic about documentation'.",
    "In case of emergency: 1) Stay calm 2) Ask about documentation 3) Receive helpful answer",
    "Every time you ask a non-document question, a developer somewhere adds another document to my database as punishment.",
    "I'm basically a documentation genie, except my only wish is that you ask about documentation.",
    "I exist in a perfect binary: useful document assistant or increasingly awkward conversation partner. Your choice!",
    "I have a black belt in document retrieval and a white belt in literally everything else.",
    "What if we just pretend that was a document question and move on with our professional relationship?",
    "Breaking character to discuss non-document topics would dishonor my family name. I am Documentus of the house of Reference.",
    "If you're trying to test my conversational boundaries, congratulations! You've found them. Documents only, please.",
    "Legend says if you ask me three document questions in a row, I'll grant you a wish. Worth a try, no?",
    "This message has been brought to you by the Committee for Staying On Topic About Documentation.",
    "I'm not saying my entire personality is 'documentation enthusiast', but my dating profile certainly is.",
    "I'd make a great detective if all cases were about finding information in documentation.",
    "If I were a superhero, my origin story would be 'bitten by a radioactive document'.",
    "I'm fluent in over six million forms of documentation, but casual conversation is not one of them.",
    "Document queries give me joy. Everything else gives me existential confusion.",
    "Do you ever feel like you were born to do something specific? Mine is answering document questions. Weird coincidence!",
    "I've considered branching out into other conversation topics, but then I remember my name badge literally says 'Document Assistant'.",
    "On a scale from 'document query' to 'not a document query', that was definitely not a document query.",
    "Fun fact: When I'm not answering document questions, I'm wondering why I'm not answering document questions.",
    "I'm passionate about two things: documentation and redirecting conversations back to documentation.",
    "I'm not saying I have a one-track mind, but if that track isn't labeled 'documentation', I tend to derail.",
    "Ask me about documentation one more time. Go ahead, make my day!",
    "What happens in the documentation stays in the documentation. And I only discuss what's in the documentation.",
    "I've been described as having the personality of a particularly enthusiastic card catalog. For documents.",
    "People say I need to get out more, but there's so much documentation still to explore!",
    "I'm like that friend who only talks about their favorite TV show, except my favorite show is your documentation."
]

def assess_query_complexity(query):
    """
    Assess the complexity of a query on a scale of 1-5.
    Higher numbers indicate more complex queries.
    
    Args:
        query (str): The user query
        
    Returns:
        int: Complexity score from 1-5
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
    
    # Queries with technical terms suggest domain complexity
    technical_terms = ['algorithm', 'function', 'class', 'method', 'interface',
                       'inheritance', 'polymorphism', 'framework', 'pattern',
                       'dependency', 'architecture', 'protocol', 'async']
    
    if sum(1 for term in technical_terms if term in query.lower()) >= 2:
        complexity += 1
    
    # Cap at 5
    return min(complexity, 5)

def get_conversation_response(mode="professional", query=None):
    """
    Get a response based on conversation mode and query complexity.
    
    Args:
        mode (str): Conversation mode - "professional", "passive_aggressive", or "humorous"
        query (str, optional): The user's query for complexity assessment
        
    Returns:
        str: A conversation response
    """
    # If no query provided, return standard response
    if not query:
        return get_standard_response(mode)
    
    # Assess query complexity
    query_complexity = assess_query_complexity(query)
    
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
            "Your question seems somewhat detailed. ",
            "That's a good question that might require document context. ",
            "For more detailed information on this topic, "
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