"""JSON serialization utilities for caching."""

import hashlib
import json
from typing import Any


def cache_json_serializer(obj: Any) -> str:
    """Serialize objects to JSON string for cache keys."""
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        # Fallback to string representation for non-serializable objects
        return str(obj)


def normalize_cache_key(key: str) -> str:
    """Normalize cache key to ensure consistency."""
    # Remove leading/trailing whitespace and convert to lowercase
    key = key.strip().lower()

    # Replace problematic characters
    key = key.replace(" ", "_")
    key = key.replace(":", "_")
    key = key.replace("/", "_")
    key = key.replace("\\", "_")

    # Ensure key length is reasonable (diskcache has limits)
    if len(key) > 250:
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        key = key[:240] + "_" + hash_suffix

    return key


def serialize_function_args(func_name: str, args: tuple, kwargs: dict) -> str:
    """Serialize function arguments for cache key generation."""
    args_str = cache_json_serializer(args)
    kwargs_str = cache_json_serializer(kwargs)

    combined = f"{func_name}({args_str},{kwargs_str})"
    return normalize_cache_key(combined)
