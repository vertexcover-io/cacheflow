"""Utility functions for caching."""

import asyncio
import functools
import hashlib
import inspect
from collections.abc import Callable
from typing import Any

from .serializers import serialize_function_args


def is_async_function(func: Callable) -> bool:
    """Check if a function is async."""
    return inspect.iscoroutinefunction(func)


def generate_cache_key(
    func_name: str, args: tuple, kwargs: dict, namespace: str | None = None
) -> str:
    """Generate a cache key from function name and arguments."""
    key = serialize_function_args(func_name, args, kwargs)

    if namespace:
        key = f"{namespace}:{key}"

    return key


def md5_hash(text: str) -> str:
    """Generate MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()


def extract_cache_control_from_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Extract cache control parameters from kwargs."""
    cache_control = {}
    filtered_kwargs = {}

    # Cache control parameters
    cache_control_keys = {"no_cache", "ttl", "namespace", "cache_key", "cache_name"}

    for key, value in kwargs.items():
        if key in cache_control_keys:
            cache_control[key] = value
        elif key == "cache_control" and isinstance(value, dict):
            cache_control.update(value)
        else:
            filtered_kwargs[key] = value

    return cache_control, filtered_kwargs


def safe_call_async(func: Callable, *args, **kwargs) -> Any:
    """Safely call an async function, handling sync/async context."""
    if is_async_function(func):
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an async context, await the coroutine
            return func(*args, **kwargs)
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(func(*args, **kwargs))
    else:
        return func(*args, **kwargs)


def create_wrapper_function(
    original_func: Callable, cache_func: Callable, preserve_signature: bool = True
) -> Callable:
    """Create a wrapper function that preserves metadata."""
    if preserve_signature:

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            return cache_func(*args, **kwargs)

        # Preserve async nature
        if is_async_function(original_func):

            @functools.wraps(original_func)
            async def async_wrapper(*args, **kwargs):
                return await cache_func(*args, **kwargs)

            return async_wrapper

        return wrapper
    else:
        return cache_func
