# MindNest Development Documentation

This directory contains technical documentation for developers working on MindNest. These documents provide details about the internal workings, architecture, and implementation details of specific features.

## Available Documentation

### Query Classification System
- [Query Classification](query_classification.md) - Overview of the query classification system
- [Query Classification Feedback](query_classification_feedback.md) - Details about the feedback learning system
- [Query Processing](query_processing.md) - Information about query processing and optimization

### Performance Features
- [Context Window](context_window.md) - Documentation on context window optimization
- [Model Support](model_support.md) - Details about model integration and capabilities

## Purpose

These documents are intended for developers working on MindNest and are not directly used by the AI to answer user queries. They provide technical details, implementation notes, and architectural information to help maintain and extend the system.

## Content Guidelines

Documents in this folder should:
- Focus on technical details relevant to developers
- Include implementation details, design patterns, and architectural decisions
- Document non-obvious behavior and design trade-offs
- Provide enough context for new developers to understand the system

## How to Add New Documentation

When implementing new features or making significant changes to the codebase:

1. Create a new Markdown file in this directory
2. Add a link to the new document in this index file
3. Follow the established format and style
4. Include code examples where appropriate
5. Document design decisions and trade-offs

## Code Standards

All code mentioned in these documents should follow the project's coding standards:
- PEP 8 compliance for Python code
- Type annotations for all function parameters and return values
- Comprehensive docstrings for all functions and classes 