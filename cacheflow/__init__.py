"""CacheFlow - Development-First Caching Library with LLM Support."""

__version__ = "0.1.0"

# Core caching functionality
# Backends (for advanced usage)
from .backends import CacheBackend, DiskCacheBackend, MemoryCacheBackend

# Configuration
from .config import configure, get_config, reset_config
from .core import (
    cache,
    cache_exists,
    cache_stats,
    clear_all_cache,
    clear_namespace,
    delete_cache_key,
)

# LLM-specific caching
from .llm import (
    llm_cache,
    replace_image_urls_with_keys,
)

# Utilities (for advanced usage)
from .serializers import cache_json_serializer, normalize_cache_key
from .utils import generate_cache_key, is_async_function

__all__ = [
    # Backends
    "CacheBackend",
    "DiskCacheBackend",
    "MemoryCacheBackend",
    # Core
    "cache",
    "cache_exists",
    # Utilities
    "cache_json_serializer",
    "cache_stats",
    "clear_all_cache",
    "clear_namespace",
    # Configuration
    "configure",
    "delete_cache_key",
    "generate_cache_key",
    "get_config",
    "is_async_function",
    # LLM
    "llm_cache",
    "normalize_cache_key",
    "replace_image_urls_with_keys",
    "reset_config",
]
