"""Core caching decorators and functions."""

import functools
import logging
from collections.abc import Callable
from typing import Any

from .config import create_backend, get_config
from .utils import (
    extract_cache_control_from_kwargs,
    generate_cache_key,
    is_async_function,
)

logger = logging.getLogger(__name__)


def cache(
    func: Callable | None = None,
    *,
    ttl: int | None = None,
    namespace: str | None = None,
    name: str | None = None,
    name_fn: Callable | None = None,
    key_fn: Callable | None = None,
    backend: str | None = None,
    enabled: bool | None = None,
):
    """
    Main caching decorator with full feature set.

    Args:
        func: Function to cache (when used without parentheses)
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        name: Custom cache name
        name_fn: Function to generate cache name from args/kwargs
        key_fn: Custom key generation function
        backend: Backend to use (overrides global config)
        enabled: Whether caching is enabled (overrides global config)
    """

    def decorator(f: Callable) -> Callable:
        config = get_config()

        # Determine if caching is enabled
        cache_enabled = enabled if enabled is not None else config.enabled

        if not cache_enabled:
            return f

        # Get the appropriate backend
        cache_backend = create_backend(backend) if backend else config.get_backend()

        # Generate function name for cache key
        func_name = name or f.__name__
        if name_fn:
            # name_fn will be called with args and kwargs at runtime
            func_name = None

        # Determine namespace
        cache_namespace = namespace or config.namespace

        # Determine TTL
        cache_ttl = ttl or config.default_ttl

        if is_async_function(f):

            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                return await _cache_function_call_async(
                    f,
                    args,
                    kwargs,
                    cache_backend,
                    func_name,
                    name_fn,
                    cache_namespace,
                    cache_ttl,
                    key_fn,
                    config.debug,
                )

            return async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                return _cache_function_call(
                    f,
                    args,
                    kwargs,
                    cache_backend,
                    func_name,
                    name_fn,
                    cache_namespace,
                    cache_ttl,
                    key_fn,
                    config.debug,
                )

            return sync_wrapper

    # Handle case where decorator is used without parentheses
    if func is not None:
        return decorator(func)

    return decorator


def _cache_function_call(
    func: Callable,
    args: tuple,
    kwargs: dict,
    cache_backend: Any,
    func_name: str | None,
    name_fn: Callable | None,
    namespace: str | None,
    ttl: int | None,
    key_fn: Callable | None,
    debug: bool,
) -> Any:
    """Handle the actual caching logic for function calls."""
    # Extract cache control parameters
    cache_control, filtered_kwargs = extract_cache_control_from_kwargs(kwargs)

    # Check for no_cache override
    if cache_control.get("no_cache", False):
        if debug:
            logger.info(f"Cache disabled for {func.__name__}")
        return func(*args, **filtered_kwargs)

    # Override parameters from cache_control
    actual_ttl = cache_control.get("ttl", ttl)
    actual_namespace = cache_control.get("namespace", namespace)
    actual_name = cache_control.get("cache_name", func_name)

    # Generate cache key
    if key_fn:
        cache_key = key_fn(func.__name__, args, filtered_kwargs)
    else:
        if name_fn:
            actual_name = name_fn(args, filtered_kwargs)
        elif actual_name is None:
            actual_name = func.__name__

        cache_key = generate_cache_key(
            actual_name, args, filtered_kwargs, actual_namespace
        )

    # Check if custom cache key provided
    if "cache_key" in cache_control:
        cache_key = cache_control["cache_key"]

    # Try to get from cache
    cached_result = cache_backend.get(cache_key)
    if cached_result is not None:
        if debug:
            logger.info(f"Cache hit for {func.__name__}: {cache_key}")
        return cached_result

    # Execute function
    if debug:
        logger.info(f"Cache miss for {func.__name__}: {cache_key}")

    result = func(*args, **filtered_kwargs)

    # Store in cache
    cache_backend.set(cache_key, result, actual_ttl)

    if debug:
        logger.info(f"Cached result for {func.__name__}: {cache_key}")

    return result


async def _cache_function_call_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    cache_backend: Any,
    func_name: str | None,
    name_fn: Callable | None,
    namespace: str | None,
    ttl: int | None,
    key_fn: Callable | None,
    debug: bool,
) -> Any:
    """Handle the actual caching logic for async function calls."""
    # Extract cache control parameters
    cache_control, filtered_kwargs = extract_cache_control_from_kwargs(kwargs)

    # Check for no_cache override
    if cache_control.get("no_cache", False):
        if debug:
            logger.info(f"Cache disabled for {func.__name__}")
        return await func(*args, **filtered_kwargs)

    # Override parameters from cache_control
    actual_ttl = cache_control.get("ttl", ttl)
    actual_namespace = cache_control.get("namespace", namespace)
    actual_name = cache_control.get("cache_name", func_name)

    # Generate cache key
    if key_fn:
        cache_key = key_fn(func.__name__, args, filtered_kwargs)
    else:
        if name_fn:
            actual_name = name_fn(args, filtered_kwargs)
        elif actual_name is None:
            actual_name = func.__name__

        cache_key = generate_cache_key(
            actual_name, args, filtered_kwargs, actual_namespace
        )

    # Check if custom cache key provided
    if "cache_key" in cache_control:
        cache_key = cache_control["cache_key"]

    # Try to get from cache
    cached_result = cache_backend.get(cache_key)
    if cached_result is not None:
        if debug:
            logger.info(f"Cache hit for {func.__name__}: {cache_key}")
        return cached_result

    # Execute function
    if debug:
        logger.info(f"Cache miss for {func.__name__}: {cache_key}")

    result = await func(*args, **filtered_kwargs)

    # Store in cache
    cache_backend.set(cache_key, result, actual_ttl)

    if debug:
        logger.info(f"Cached result for {func.__name__}: {cache_key}")

    return result


def delete_cache_key(key: str, namespace: str | None = None) -> bool:
    """Delete specific cache entry."""
    config = get_config()
    cache_backend = config.get_backend()

    if namespace:
        key = f"{namespace}:{key}"

    return cache_backend.delete(key)


def clear_namespace(namespace: str) -> int:
    """Clear all entries in namespace."""
    config = get_config()
    cache_backend = config.get_backend()

    return cache_backend.clear_namespace(namespace)


def cache_stats() -> dict[str, Any]:
    """Get cache hit/miss statistics."""
    config = get_config()
    cache_backend = config.get_backend()

    return cache_backend.get_stats()


def cache_exists(key: str, namespace: str | None = None) -> bool:
    """Check if cache key exists."""
    config = get_config()
    cache_backend = config.get_backend()

    if namespace:
        key = f"{namespace}:{key}"

    return cache_backend.exists(key)


def get_cache_key(
    func: Callable, args: tuple, kwargs: dict, namespace: str | None = None
) -> str:
    """Get cache key for debugging purposes."""
    # Extract cache control parameters
    cache_control, filtered_kwargs = extract_cache_control_from_kwargs(kwargs)

    # Use provided namespace or default
    actual_namespace = cache_control.get("namespace", namespace)

    return generate_cache_key(func.__name__, args, filtered_kwargs, actual_namespace)


def clear_all_cache() -> None:
    """Clear all cache entries."""
    config = get_config()
    cache_backend = config.get_backend()

    cache_backend.clear()


def warm_cache(func: Callable, param_sets: list[tuple]) -> None:
    """Pre-populate cache with known values."""
    if not is_async_function(func):
        for params in param_sets:
            if isinstance(params, tuple):
                args, kwargs = params if len(params) == 2 else (params, {})
            else:
                args, kwargs = (params,), {}

            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to warm cache for {func.__name__}: {e}")
    else:
        logger.warning("Async function cache warming not supported in sync context")
