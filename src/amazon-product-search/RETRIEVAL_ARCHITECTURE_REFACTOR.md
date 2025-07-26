# Retrieval Architecture Refactor

This document outlines the comprehensive refactor of the retrieval system in `amazon_product_search` to address organizational issues regarding abstraction levels and flexibility.

## Problems Addressed

### 1. Mixed and Inconsistent Abstraction Levels
- **Before**: The `Retriever` class mixed high-level orchestration with low-level implementation details
- **After**: Clear separation of concerns with dedicated abstractions for each responsibility

### 2. Tight Coupling to Infrastructure  
- **Before**: Heavy coupling to Elasticsearch, making it difficult to extend or replace
- **After**: Modular engines that can be easily swapped or extended

### 3. Poor Integration Between Components
- **Before**: Rerankers, fusion, and retrieval existed in isolation
- **After**: Unified pipeline that integrates all components seamlessly

### 4. Inflexible Fusion Strategy
- **Before**: Hardcoded fusion for exactly two retrieval methods (lexical + semantic)
- **After**: Flexible fusion system supporting any number of retrieval engines

### 5. Code Duplication and Resource Waste
- **Before**: Multiple components implemented similar patterns independently
- **After**: Shared resource manager eliminates duplication

### 6. Overly Complex Central Classes
- **Before**: Large classes handling multiple concerns
- **After**: Focused, single-responsibility components

## New Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Query           │    │ Retrieval        │    │ Result          │
│ Processors      │───▶│ Engines          │───▶│ Fusion          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ • Normalization │    │ • Lexical Engine │    │ • Weighted Sum  │
│ • Tokenization  │    │ • Semantic Engine│    │ • RRF           │
│ • Vector Encoding│    │ • Custom Engines │    │ • Borda Count   │
│ • Synonym Expand│    │ • Multi-Strategy │    │ • Max Score     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                            ┌─────────────────┐
                                            │ Post            │
                                            │ Processors      │
                                            └─────────────────┘
                                                    │
                                                    ▼
                                            ┌─────────────────┐
                                            │ • Rerankers     │
                                            │ • Filters       │
                                            │ • Custom Logic  │
                                            └─────────────────┘
```

## Key Components

### Core Abstractions (`retrieval/core/`)
- **Protocols**: Define interfaces for all major components
- **Types**: Shared data structures and enums
- **Clean separation** between interfaces and implementations

### Query Processors (`retrieval/processors/`)
- **BaseQueryProcessor**: Handles normalization and tokenization
- **SynonymExpandingProcessor**: Adds synonym expansion
- **SemanticQueryProcessor**: Adds vector encoding
- **ProcessorChain**: Combines multiple processors

### Retrieval Engines (`retrieval/engines/`)
- **BaseRetrievalEngine**: Common functionality
- **LexicalRetrievalEngine**: BM25/keyword search
- **SemanticRetrievalEngine**: Dense vector similarity
- **Extensible**: Easy to add new engine types

### Result Fusion (`retrieval/fusion/`)
- **FlexibleResultFuser**: Supports multiple fusion methods
- **Configurable**: Weighted sum, RRF, Borda count, max score
- **Scalable**: Can fuse any number of engine responses

### Resource Management (`retrieval/resources/`)
- **SharedResourceManager**: Singleton pattern for shared resources
- **Model caching**: Eliminates duplicate model loading
- **Memory efficient**: Proper cleanup and lifecycle management

### Pipeline Orchestration (`retrieval/pipeline.py`)
- **RetrievalPipeline**: Orchestrates the entire retrieval flow
- **Modular**: Easy to add/remove components
- **Observable**: Rich metadata and logging

## Backward Compatibility

The refactored `Retriever` class maintains complete backward compatibility:

```python
# Old API still works exactly the same
retriever = Retriever(locale="jp")
response = retriever.search(
    index_name="products_jp",
    query="wireless headphones", 
    fields=["product_title", "title_vector"],
    lexical_boost=0.7,
    semantic_boost=1.3
)
```

But now also supports the new architecture:

```python
# New modular approach
from amazon_product_search.retrieval.factory import create_retrieval_system

pipeline = create_retrieval_system("hybrid_rrf", locale="jp")
response = pipeline.search(raw_query, config)
```

## Benefits

### 1. **Flexibility**
- Easy to add new retrieval methods
- Configurable fusion strategies
- Pluggable post-processors

### 2. **Maintainability**
- Clear separation of concerns
- Single-responsibility components
- Proper abstractions

### 3. **Performance**
- Shared resource management
- No duplicate model loading
- Efficient memory usage

### 4. **Extensibility**
- Protocol-based design allows easy extensions
- Factory functions simplify configuration
- Plugin architecture for custom components

### 5. **Testability**
- Each component can be tested in isolation
- Clear interfaces make mocking easy
- Dependency injection support

## Migration Guide

### For Existing Code
No changes needed - backward compatibility is maintained.

### For New Development
Use the factory functions for easy setup:

```python
from amazon_product_search.retrieval.factory import create_retriever_with_new_architecture

# Basic usage
retriever = create_retriever_with_new_architecture(locale="jp")

# Advanced usage with custom configuration
pipeline = create_advanced_retrieval_pipeline(
    locale="jp",
    fusion_method="rrf", 
    reranker_type="colbert",
    enable_filtering=True
)
```

### Adding Custom Components

```python
# Custom retrieval engine
class MyCustomEngine(BaseRetrievalEngine):
    def retrieve(self, query, config):
        # Custom retrieval logic
        pass

# Add to pipeline
retriever.add_retrieval_engine(MyCustomEngine())

# Custom post-processor
class MyFilter(ResultProcessor):
    def process(self, response, query, config):
        # Custom processing logic
        pass

retriever.add_post_processor(MyFilter())
```

## Example Configurations

### Basic Hybrid Search
```python
system = create_retrieval_system("basic", locale="jp")
```

### Advanced with ColBERT Reranking
```python
system = create_retrieval_system("colbert_advanced", locale="jp")
```

### Custom Configuration
```python
pipeline = create_advanced_retrieval_pipeline(
    locale="jp",
    fusion_method="weighted_sum",
    reranker_type="dot",
    enable_filtering=True
)
```

## Future Enhancements

The new architecture enables easy implementation of:

1. **Multi-stage retrieval** (candidate generation + reranking)
2. **Learning-to-rank** integration
3. **A/B testing** of different strategies
4. **Real-time model updates**
5. **Distributed retrieval** across multiple indices
6. **Custom scoring functions**
7. **Query understanding** components

## Conclusion

This refactor transforms the retrieval system from a monolithic, tightly-coupled design into a modular, flexible, and maintainable architecture while maintaining full backward compatibility. The new system is easier to extend, test, and optimize, providing a solid foundation for future enhancements. 