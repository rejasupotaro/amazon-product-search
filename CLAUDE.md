# Amazon Product Search - Claude Code Instructions

## Project Overview
This is an Amazon product search system that showcases and compares various search algorithms and models using the Shopping Queries Dataset. The project includes:

- Search engines (Elasticsearch, Vespa)
- Dense retrieval models
- Indexing pipelines
- Model serving infrastructure
- Training pipelines
- Demo applications

## Development Workflow

### Linting and Type Checking
After making any code changes, always run:
```bash
make lint
```

This command runs:
- `ruff check --fix --unsafe-fixes --show-fixes` for code formatting and linting
- `mypy` type checking across all project modules

### Project Structure
The project uses a monorepo structure with multiple components in `src/`:
- `amazon-product-search/` - Core search functionality
- `data-source/` - Data loading utilities
- `dense-retrieval/` - Dense retrieval models and training
- `indexing/` - Data indexing pipelines
- `model-serving/` - Model serving infrastructure
- `training/` - Model training pipelines
- `demo/` - Demo applications

### Python Environment
- Python 3.11.8
- Poetry for dependency management
- Each component has its own `pyproject.toml` and dependencies