# Contributing to ColiVara Document Q&A

First off, thank you for considering contributing to ColiVara Document Q&A! It's people like you that make this project better.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, welcoming, and considerate of others.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what you expected to see**
- **Include screenshots if possible**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List some examples of how it would be used**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the coding standards below
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Commit your changes** with clear commit messages
6. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/colivara-autodeploy.git
cd colivara-autodeploy

# Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Maximum line length: 100 characters

### Code Formatting

We use `black` for code formatting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Linting

We use `flake8` for linting:

```bash
# Run linter
flake8 --max-line-length=100 --exclude=venv
```

### Type Hints

Use type hints for function parameters and return values:

```python
def process_document(file_path: str, collection_name: str) -> dict:
    """Process and upload a document.
    
    Args:
        file_path: Path to the document file
        collection_name: Name of the collection to add to
        
    Returns:
        Dictionary containing upload status and metadata
    """
    # Implementation
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Write tests for all new features
- Ensure tests are deterministic
- Use descriptive test names
- Test edge cases and error conditions

Example:

```python
def test_upload_document_success():
    """Test successful document upload"""
    # Arrange
    file_path = "test_document.pdf"
    collection = "test_collection"
    
    # Act
    result = upload_document(file_path, collection)
    
    # Assert
    assert result["success"] is True
    assert "document_id" in result
```

## Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Examples:

```
Add RAG/LLM toggle feature

- Implement toggle switch in UI
- Add backend endpoint for pure LLM queries
- Update settings to persist toggle state

Closes #123
```

## Branch Naming

Use descriptive branch names with prefixes:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or updates

Examples:
- `feature/add-streaming-responses`
- `fix/document-upload-error`
- `docs/update-installation-guide`

## Project Structure

When adding new features, follow the existing project structure:

```
colivara-autodeploy/
├── api/              # FastAPI modules
│   ├── routers/     # API routes
│   ├── services/    # Business logic
│   └── schemas/     # Pydantic models
├── routes/          # Flask blueprints
├── templates/       # HTML templates
├── models/          # Database models
├── services/        # Shared services
├── middleware/      # Middleware
└── utils/           # Utility functions
```

## Documentation

- Update README.md if you change functionality
- Add docstrings to all new functions and classes
- Update API documentation for new endpoints
- Include examples in documentation

## Review Process

1. **Automated checks** will run on your PR (linting, tests)
2. **Code review** by maintainers
3. **Testing** by maintainers if needed
4. **Approval and merge** once all checks pass

## Questions?

Feel free to:
- Open an issue for questions
- Join our discussions
- Contact the maintainers

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing! 🎉
