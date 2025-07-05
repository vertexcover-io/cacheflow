# CacheFlow

A development-first Python caching library with specialized LLM support. CacheFlow provides decorators for caching expensive operations, with particular optimization for LLM API calls that include image content.

## Features

- **Pure Function Approach**: Simple decorators over complex classes
- **Async/Sync Compatibility**: Single decorators handle both async and sync functions
- **LLM Optimization**: Specialized caching for LLM API calls with image normalization
- **Pluggable Backends**: Support for disk and memory backends
- **Image Content Handling**: Smart caching for LLM calls with image content

## Installation

```bash
pip install cacheflow
```

For development:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### Basic Caching

```python
from cacheflow import cache

@cache
def expensive_operation(x, y):
    # Expensive computation
    return x * y

# Works with async functions too
@cache
async def async_operation(data):
    # Async operation
    return await process_data(data)
```

### LLM Caching

```python
from cacheflow import llm_cache

@llm_cache(provider="openai")
def chat_completion(messages):
    # OpenAI API call
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

## Configuration

Configure CacheFlow using environment variables:

```bash
export CACHEFLOW_ENABLED=true
export CACHEFLOW_BACKEND=disk
export CACHEFLOW_CACHE_DIR=/path/to/cache
export CACHEFLOW_DEBUG=false
```

## Supported LLM Providers

- **OpenAI**: Handles `image_url` content in messages
- **Anthropic**: Processes `image` content with base64 data
- **Gemini**: Normalizes `inline_data` parts

## Backends

- **Disk Backend**: Persistent caching to filesystem
- **Memory Backend**: In-memory caching for fast access

## Development

### Running Tests

```bash
pytest
pytest --cov=cacheflow
```

### Code Quality

```bash
ruff check .
ruff format .
pre-commit run --all-files
```

## Architecture

### Core Components

- **Backend System**: Pluggable cache backends
- **Core Engine**: Main caching decorators and functions
- **LLM Support**: LLM-specific caching with image normalization
- **Configuration**: Global configuration management
- **Serializers**: JSON serialization utilities

### Key Design Patterns

- Pure function decorators for simplicity
- Async/sync compatibility through single decorators
- Image content normalization for consistent cache keys
- Pluggable backend architecture for extensibility

## License

[License information]

## Contributing

[Contributing guidelines]
