"""JSON serialization utilities for caching."""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel  # type: ignore
except ImportError:
    BaseModel = None


def json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer that handles:
    - Pydantic BaseModel objects
    - Lists of BaseModel objects
    - Dicts containing BaseModel objects
    - datetime objects
    - Type objects (resolves circular references)
    """
    if BaseModel is not None and isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")

    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, type):
        # Handle all types uniformly to prevent circular references
        return f"<Type:{obj.__module__}.{obj.__name__}>"
    raise TypeError(f"Cannot serialize object {obj} of type {type(obj)}")


def cache_json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for caching that ignores bytes fields and delegates
    to json_serializer for all other cases.
    """
    if isinstance(obj, bytes):
        return None

    try:
        return json_serializer(obj)
    except (ValueError, TypeError, RecursionError):
        logging.warning(
            f"Failed to serialize object {obj} of type {type(obj)} during disk cache serialization"
        )
        return str(obj)


def normalize_cache_key(key: str) -> str:
    """Normalize cache key to ensure consistency."""
    # Remove leading/trailing whitespace and convert to lowercase
    key = key.strip().lower()

    # Replace problematic characters
    key = key.replace(" ", "_")
    key = key.replace("\t", "_")  # Handle tabs
    key = key.replace("\n", "_")  # Handle newlines
    key = key.replace("\r", "_")  # Handle carriage returns
    key = key.replace(":", "_")
    key = key.replace("/", "_")
    key = key.replace("\\", "_")
    key = key.replace(".", "_")  # Handle dots

    return key


def default_key_fn(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Generate cache key using custom format."""
    json_key = {
        "args": args,
        "kwargs": kwargs,
    }
    try:
        # Sort keys to ensure order independence
        json_str = json.dumps(json_key, default=cache_json_serializer, sort_keys=True)
    except (ValueError, TypeError, RecursionError) as e:
        # Handle circular references and other serialization errors
        logging.warning(
            f"Failed to serialize arguments for cache key generation: {e}. "
            f"Using fallback string representation."
        )
        # Fallback to string representation
        fallback_str = str(json_key)
        json_str = json.dumps({"fallback": fallback_str}, sort_keys=True)

    return hashlib.md5(json_str.encode()).hexdigest()
