"""Utility functions for caching."""

import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any

from .serializers import default_key_fn


def is_async_function(func: Callable) -> bool:
    """Check if a function is async."""
    return inspect.iscoroutinefunction(func)


def generate_cache_key(
    func_name: str,
    args: tuple,
    kwargs: dict,
    namespace: str | None = None,
    key_fn: Callable | None = None,
) -> str:
    """Generate a cache key from function name and arguments."""
    # Generate hash from function arguments
    hash_part = key_fn(args, kwargs) if key_fn else default_key_fn(args, kwargs)

    # Format as namespace:name:hash
    if namespace:
        key = f"{namespace}:{func_name}:{hash_part}"
    else:
        key = f"{func_name}:{hash_part}"

    return key


def extract_cache_control_from_kwargs(kwargs: dict) -> tuple[bool, dict]:
    """Extract cache control parameters from kwargs."""
    filtered_kwargs = {**kwargs}
    skip = filtered_kwargs.pop("_cacheflow_skip", False)
    return skip, filtered_kwargs


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
