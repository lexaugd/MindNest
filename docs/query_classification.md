# Query Classification System

## Overview

The MindNest query classification system categorizes user queries into different types to provide the most appropriate response. The system uses a zero-shot classification approach based on category descriptions rather than hardcoded examples.

## Classification Categories

The system classifies queries into the following categories:

- **CONVERSATION**: General conversational queries, greetings, acknowledgments, or personal questions.
- **DOCUMENT_QUERY**: Questions seeking information from documents, explanations, or details about topics.
- **CONCISE_QUERY**: Requests for brief or summarized information, quick explanations, or overview.
- **DOCUMENT_SEARCH**: Explicit search-like queries to find specific information or documents.

## Components

### ZeroShotClassifier

The core of the classification system is the `ZeroShotClassifier` which uses semantic similarity between the query and category descriptions to determine the most appropriate category. It calculates cosine similarity between the query embedding and category description embeddings.

Benefits:
- No hardcoded examples required
- Can learn from user feedback over time
- Adapts to the specific application domain through feedback

### FeedbackCollector

The `FeedbackCollector` component collects and stores user feedback on classification results. This feedback is used to improve the classifier over time.

Features:
- Records classification feedback (correct/incorrect)
- Tracks confidence scores and accuracy
- Provides statistics on classification performance
- Supports training the classifier with validated feedback

## Integration with API

The system integrates with the MindNest API through:

1. **Main query endpoint** (`/ask`): Automatically classifies queries and can collect implicit feedback
2. **Explicit feedback endpoint** (`/feedback/classification`): Allows users to provide direct feedback
3. **Training endpoint** (`/classifier/train`): Uses collected feedback to improve the classifier

## Feedback-Based Learning

The system can learn from user feedback in several ways:

1. When users provide explicit feedback, it's stored in the feedback database
2. Periodically, the system can use high-confidence, correct classifications to add new descriptions
3. The training process ensures only high-quality feedback is used for learning

## Implementation Details

### Description Generation

When learning from feedback, the system automatically generates descriptive category descriptions based on the query patterns, using the `generate_description_from_query` method. This creates more meaningful descriptions than simply using the raw queries.

### Similarity Threshold

A similarity threshold (default: 0.65) determines when a classification is considered confident. If no category exceeds this threshold, the system defaults to `DOCUMENT_QUERY` as the safest option.

## Usage Example

```python
from mindnest.utils.query_classification.classifiers.zero_shot_classifier import ZeroShotClassifier
from mindnest.utils.query_classification.feedback import FeedbackCollector

# Initialize classifier
classifier = ZeroShotClassifier(embeddings_model)
classifier.initialize()

# Classify a query
category, confidence = classifier.classify("How does the authentication system work?")

# Record feedback
feedback_collector = FeedbackCollector()
feedback_collector.add_feedback(
    query="How does the authentication system work?",
    predicted_category=category,
    confidence=confidence,
    is_correct=True
)

# Train with feedback
classifier.train_from_feedback(feedback_collector.get_feedback_entries())
``` 