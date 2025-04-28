# MindNest Documentation Structure

This directory contains documentation for the MindNest project, organized into a clear structure to separate AI reference documentation from developer-focused technical documentation.

## Directory Structure

- **[ai/](ai/)** - Documentation intended for AI reference
  - Contains documents that the MindNest AI will use to answer user queries
  - Focus on end-user information, usage guides, and feature explanations
  
- **[development/](development/)** - Technical documentation for developers
  - Contains details about system architecture, implementation, and design decisions
  - Focus on code structure, algorithms, and technical details

## Documentation Organization Guidelines

### AI Reference Documentation (`ai/`)

Files in this directory should:
- Be focused on information that would help users understand and use the system
- Avoid implementation details unless relevant to users
- Be written in clear, non-technical language where possible
- Include practical examples and use cases

### Development Documentation (`development/`)

Files in this directory should:
- Focus on technical implementation details
- Include code examples and algorithm explanations
- Document architectural decisions and trade-offs
- Provide context for developers working on the system

## Legacy Documentation

Some documentation files remain in their original locations for backward compatibility. These will gradually be migrated to the new structure or removed if redundant.

## Contributing Documentation

When adding new documentation:

1. Determine whether the content is primarily for AI reference or for developers
2. Place the file in the appropriate directory (`ai/` or `development/`)
3. Update the relevant index file with a link to your new document
4. Follow the established format and style guidelines
5. Include metadata like author, date, and version where appropriate

## Updating the Documentation Structure

This documentation structure is designed to be maintainable and scalable. As the project evolves, the structure may be refined to better serve the needs of both users and developers. 