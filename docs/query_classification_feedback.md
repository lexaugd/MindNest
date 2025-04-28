# Query Classification Feedback Learning

This document describes the feedback-based learning system for query classification in MindNest.

## Overview

The feedback learning system allows the query classifier to improve over time based on user feedback. When a user provides feedback about misclassified queries, the system can learn from these mistakes and adapt the classifier to improve future classifications.

## How It Works

1. **Feedback Collection**: When users interact with the system, they can provide feedback about query classifications through the `/feedback/classification` endpoint.

2. **Learning Process**: The `FeedbackLearner` periodically analyzes collected feedback and identifies high-confidence misclassifications.

3. **Classifier Update**: The learner then updates the classifier's examples with the correctly labeled queries, allowing the classifier to learn from its mistakes.

4. **Cache Clearing**: After updating the examples, any classification caches are cleared to ensure the new knowledge is immediately available.

## Key Components

### FeedbackCollector

The `FeedbackCollector` class stores and organizes user feedback on query classifications. It maintains statistics on accuracy and confidence levels.

```python
# Example usage
from mindnest.utils.query_classification.feedback import FeedbackCollector

feedback_collector = FeedbackCollector()
feedback_collector.add_feedback(
    query="How does the system work?",
    predicted_category="CONVERSATION",
    correct_category="DOCUMENT_QUERY",
    confidence=0.85,
    is_correct=False
)
```

### FeedbackLearner

The `FeedbackLearner` class analyzes feedback data and updates the classifier examples based on misclassifications:

```python
# Example usage
from mindnest.utils.query_classification.feedback_learning import FeedbackLearner

learner = FeedbackLearner(classifier)
learner.check_and_update()  # Checks if enough new feedback is available and updates if needed
```

## API Endpoints

The following endpoints support the feedback learning system:

### Submit Feedback

```
POST /feedback/classification
```

**Request Body:**
```json
{
  "query": "How does authentication work?",
  "predicted_category": "CONVERSATION",
  "correct_category": "DOCUMENT_QUERY",
  "confidence": 0.85
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback recorded successfully",
  "stats": {
    "total_feedback": 42,
    "accuracy": 0.88,
    "correct_classifications": 37,
    "incorrect_classifications": 5
  },
  "classifier_updated": true
}
```

### Force Classifier Update

```
POST /classifier/update-from-feedback
```

**Response:**
```json
{
  "status": "success",
  "message": "Classifier updated successfully from feedback data",
  "updated": true,
  "feedback_learning_stats": {
    "updates": 3,
    "new_examples_added": 12,
    "examples_by_category": {
      "DOCUMENT_QUERY": 15,
      "CONVERSATION": 12,
      "CONCISE_QUERY": 13,
      "DOCUMENT_SEARCH": 11
    }
  }
}
```

### Get Classifier Examples

```
GET /classifier/examples
```

**Response:**
```json
{
  "status": "success",
  "examples": {
    "CONVERSATION": ["How are you?", "What's your name?", "..."],
    "DOCUMENT_QUERY": ["How does authentication work?", "..."],
    "CONCISE_QUERY": ["Summarize authentication", "..."],
    "DOCUMENT_SEARCH": ["Find authentication docs", "..."]
  },
  "category_counts": {
    "CONVERSATION": 12,
    "DOCUMENT_QUERY": 15,
    "CONCISE_QUERY": 13,
    "DOCUMENT_SEARCH": 11
  },
  "feedback_learning_enabled": true,
  "feedback_learning_stats": {
    "updates": 3,
    "new_examples_added": 12
  }
}
```

### Configure Feedback Learning

```
POST /config/model
```

**Request Body:**
```json
{
  "enable_feedback_learning": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration updated.",
  "config": {
    "use_small_model": false,
    "max_context_tokens": 800,
    "conversation_mode": "professional",
    "query_classifier_mode": "embeddings",
    "feedback_learning_enabled": true
  }
}
```

## Best Practices

1. **Threshold Control**: The `confidence_threshold` parameter (default 0.75) ensures that only high-confidence misclassifications are used for learning, reducing the risk of learning from noisy feedback.

2. **Periodic Updates**: Updates occur after collecting a specified number of feedback entries (default 50) to ensure enough data for meaningful updates.

3. **Example Limits**: The system limits the number of learned examples per category to prevent over-optimization toward specific patterns.

4. **Transparency**: The `/classifier/examples` endpoint allows developers to inspect what the classifier has learned, providing transparency into the learning process.

## Configuration

Feedback learning can be enabled or disabled through the model configuration API. When enabled, the system will automatically learn from user feedback and improve the classifier over time.

## Monitoring

The system tracks statistics about both the feedback collection and the learning process:

- **Feedback Statistics**: Accuracy, total feedback count, correct and incorrect classifications
- **Learning Statistics**: Number of updates, examples added, and examples per category

These statistics are available through the feedback and classifier API endpoints. 