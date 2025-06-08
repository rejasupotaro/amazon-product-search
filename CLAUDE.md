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
- `uv run ruff check --fix --unsafe-fixes --show-fixes` for code formatting and linting
- `uv run mypy` type checking across all project modules

### Package Management - UV Workspace
The project uses **UV workspaces** for fast, modern Python package management:

**Initial Setup:**
```bash
uv sync  # Install all workspace packages and dependencies
```

**Common Commands:**
```bash
uv sync                                    # Install/update all dependencies
uv add --package indexing pandas          # Add dependency to specific package
uv run --package demo streamlit run app.py # Run command for specific package
uv run ruff check src/                    # Run command across workspace
```

**Backward Compatibility:**
- Poetry configurations are maintained for team members still using Poetry
- Use `make lint-poetry` for Poetry-based linting if needed

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
- Python 3.11+ required
- UV for fast dependency management and workspace support
- Each component has its own `pyproject.toml` with both UV and Poetry configurations
- Unified `uv.lock` file for deterministic builds across all packages