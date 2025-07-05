"""Unit tests for core caching functionality."""

import time
from unittest.mock import patch

import pytest

from cacheflow.config import configure, get_config
from cacheflow.core import (
    _cache_function_call,
    _cache_function_call_async,
    cache,
    cache_exists,
    cache_stats,
    clear_all_cache,
    clear_namespace,
    delete_cache_key,
)
from cacheflow.utils import generate_cache_key


class TestCacheDecorator:
    """Test the main cache decorator functionality."""

    def test_cache_sync_function_basic(self, sample_function):
        """Test basic sync function caching works correctly."""
        cached_func = cache(sample_function)

        # First call should execute function
        result1 = cached_func(1, 2)
        assert result1 == 3
        assert sample_function.call_count() == 1

        # Second call should use cache
        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        # Different arguments should execute function again
        result3 = cached_func(2, 3)
        assert result3 == 5
        assert sample_function.call_count() == 2

    @pytest.mark.asyncio
    async def test_cache_async_function_basic(self, sample_async_function):
        """Test basic async function caching works correctly."""
        cached_func = cache(sample_async_function)

        # First call should execute function
        result1 = await cached_func(1, 2)
        assert result1 == 3
        assert sample_async_function.call_count() == 1

        # Second call should use cache
        result2 = await cached_func(1, 2)
        assert result2 == 3
        assert sample_async_function.call_count() == 1

        # Different arguments should execute function again
        result3 = await cached_func(2, 3)
        assert result3 == 5
        assert sample_async_function.call_count() == 2

    def test_cache_disabled_globally(self, sample_function):
        """Test cache behavior when disabled globally."""
        # Disable cache globally
        configure(enabled=False)

        cached_func = cache(sample_function)

        # Both calls should execute function
        result1 = cached_func(1, 2)
        result2 = cached_func(1, 2)

        assert result1 == 3
        assert result2 == 3
        assert sample_function.call_count() == 2

    def test_cache_disabled_per_decorator(self, sample_function):
        """Test cache disabled per decorator instance."""
        cached_func = cache(sample_function, enabled=False)

        # Both calls should execute function
        result1 = cached_func(1, 2)
        result2 = cached_func(1, 2)

        assert result1 == 3
        assert result2 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_skip_parameter(self, sample_function):
        """Test cache skip using _cacheflow_skip parameter."""
        cached_func = cache(sample_function)

        # First call - should cache
        result1 = cached_func(1, 2)
        assert result1 == 3
        assert sample_function.call_count() == 1

        # Second call - should use cache
        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        # Third call with skip - should execute function
        result3 = cached_func(1, 2, _cacheflow_skip=True)
        assert result3 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_custom_namespace(self, sample_function):
        """Test cache with custom namespace."""
        cached_func = cache(sample_function, namespace="test_ns")

        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        cached_func2 = cache(sample_function)

        result3 = cached_func2(1, 2)
        assert result3 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_custom_name(self, sample_function):
        """Test cache with custom name."""
        cached_func = cache(sample_function, name="custom_add")

        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        cached_func2 = cache(sample_function)

        result3 = cached_func2(1, 2)
        assert result3 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_name_function(self, sample_function):
        """Test cache with custom name function."""

        def custom_name_fn(args, kwargs):
            return f"sum_{args[0]}_{args[1]}"

        cached_func = cache(sample_function, name_fn=custom_name_fn)

        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        cached_func2 = cache(sample_function)

        result3 = cached_func2(1, 2)
        assert result3 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_custom_key_function(self, sample_function):
        """Test cache with custom key function."""

        def custom_key_fn(args, kwargs):
            return f"key_{args[0]}{args[1]}"

        cached_func = cache(sample_function, key_fn=custom_key_fn)

        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        # Verify custom key function is used
        backend = get_config().get_backend()
        if hasattr(backend, "_cache"):
            keys = list(backend._cache.keys())
            assert any("key_12" in key for key in keys)

    def test_cache_with_ttl_expiration(self, sample_function):
        """Test cache with TTL expiration."""
        cached_func = cache(sample_function, ttl=1)

        # First call
        result1 = cached_func(1, 2)
        assert result1 == 3
        assert sample_function.call_count() == 1

        # Second call - should use cache
        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Third call - should execute function again
        result3 = cached_func(1, 2)
        assert result3 == 3
        assert sample_function.call_count() == 2

    def test_cache_with_function_exception(self):
        """Test cache behavior when function raises exception."""
        call_count = 0

        def failing_function(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test error")
            return "success"

        cached_func = cache(failing_function)

        # Function succeeds - should cache
        result = cached_func(False)
        assert result == "success"
        assert call_count == 1

        # Same call - should use cache
        result = cached_func(False)
        assert result == "success"
        assert call_count == 1

        # Function fails - should not cache
        with pytest.raises(ValueError, match="Test error"):
            cached_func(True)
        assert call_count == 2

        # Same failing call - should call function again
        with pytest.raises(ValueError, match="Test error"):
            cached_func(True)
        assert call_count == 3

    def test_cache_with_mutable_arguments(self):
        """Test cache with mutable arguments."""
        call_count = 0

        def process_list(items):
            nonlocal call_count
            call_count += 1
            return sum(items)

        cached_func = cache(process_list)

        # First call
        result1 = cached_func([1, 2, 3])
        assert result1 == 6
        assert call_count == 1

        # Same content - should use cache
        result2 = cached_func([1, 2, 3])
        assert result2 == 6
        assert call_count == 1

        # Different content - should call function
        result3 = cached_func([1, 2, 3, 4])
        assert result3 == 10
        assert call_count == 2

    def test_cache_without_parentheses(self, sample_function):
        """Test cache decorator used without parentheses."""

        @cache
        def add_numbers(a, b):
            return sample_function(a, b)

        result1 = add_numbers(1, 2)
        result2 = add_numbers(1, 2)

        assert result1 == 3
        assert result2 == 3
        assert sample_function.call_count() == 1

    def test_cache_with_parentheses(self, sample_function):
        """Test cache decorator used with parentheses."""

        @cache()
        def add_numbers(a, b):
            return sample_function(a, b)

        result1 = add_numbers(1, 2)
        result2 = add_numbers(1, 2)

        assert result1 == 3
        assert result2 == 3
        assert sample_function.call_count() == 1

    def test_cache_with_kwargs(self, sample_function):
        """Test cache with keyword arguments."""
        cached_func = cache(sample_function)

        # Test with kwargs
        result1 = cached_func(a=1, b=2)
        assert result1 == 3
        assert sample_function.call_count() == 1

        # Same kwargs - should use cache
        result2 = cached_func(a=1, b=2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        # Different kwargs - should call function
        result3 = cached_func(a=2, b=3)
        assert result3 == 5
        assert sample_function.call_count() == 2

    def test_cache_with_mixed_args_kwargs(self, sample_function):
        """Test cache with mixed positional and keyword arguments."""
        cached_func = cache(sample_function)

        # Test with mixed args
        result1 = cached_func(1, b=2)
        assert result1 == 3
        assert sample_function.call_count() == 1

        # Same args - should use cache
        result2 = cached_func(1, b=2)
        assert result2 == 3
        assert sample_function.call_count() == 1

        # Different argument pattern - should call function again
        result3 = cached_func(b=2, a=1)
        assert result3 == 3
        assert sample_function.call_count() == 2


class TestCacheUtilityFunctions:
    """Test utility functions for cache management."""

    def test_delete_cache_key_basic(self, sample_function):
        """Test deleting specific cache key."""
        cached_func = cache(sample_function)

        # Cache a result
        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        # Generate the same key that was used
        key = generate_cache_key("_sample_function", (1, 2), {})

        # Delete the cache key
        deleted = delete_cache_key(key)
        assert deleted

        # Verify key is deleted by calling function again
        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 2

    def test_delete_cache_key_with_namespace(self, sample_function):
        """Test deleting cache key with namespace."""
        cached_func = cache(sample_function, namespace="test_ns")

        # Cache a result
        result = cached_func(1, 2)
        assert result == 3
        assert sample_function.call_count() == 1

        # Delete with namespace
        key = generate_cache_key("_sample_function", (1, 2), {})
        deleted = delete_cache_key(key, namespace="test_ns")
        assert deleted

        # Verify deletion by calling function again
        result2 = cached_func(1, 2)
        assert result2 == 3
        assert sample_function.call_count() == 2

    def test_delete_nonexistent_key(self):
        """Test deleting non-existent cache key."""
        deleted = delete_cache_key("nonexistent_key")
        assert not deleted

    def test_clear_namespace(self, sample_function):
        """Test clearing entire namespace."""
        cached_func1 = cache(sample_function, namespace="test_ns")

        def multiply_func(a, b):
            return a * b

        cached_func2 = cache(multiply_func, namespace="test_ns")

        # Cache some results
        cached_func1(1, 2)
        cached_func2(3, 4)

        assert sample_function.call_count() == 1

        # Clear the namespace
        count = clear_namespace("test_ns")
        assert count >= 0  # Number of deleted keys

        # Verify namespace is cleared by calling functions again
        cached_func1(1, 2)
        cached_func2(3, 4)
        assert sample_function.call_count() == 2

    def test_cache_stats(self, sample_function):
        """Test getting cache statistics."""
        cached_func = cache(sample_function)

        # Make some calls
        cached_func(1, 2)  # miss
        cached_func(1, 2)  # hit
        cached_func(3, 4)  # miss

        stats = cache_stats()
        assert isinstance(stats, dict)
        # Stats format depends on backend implementation

    def test_cache_exists(self, sample_function):
        """Test checking if cache key exists."""
        cached_func = cache(sample_function)

        # Cache a result
        cached_func(1, 2)
        assert sample_function.call_count() == 1

        # Check if key exists
        key = generate_cache_key("_sample_function", (1, 2), {})
        assert cache_exists(key)

        # Check non-existent key
        key2 = generate_cache_key("_sample_function", (5, 6), {})
        assert not cache_exists(key2)

    def test_cache_exists_with_namespace(self, sample_function):
        """Test checking if cache key exists with namespace."""
        cached_func = cache(sample_function, namespace="test_ns")

        # Cache a result
        cached_func(1, 2)
        assert sample_function.call_count() == 1

        # Check if key exists with namespace
        key = generate_cache_key("_sample_function", (1, 2), {})
        assert cache_exists(key, namespace="test_ns")

        # Check without namespace - should not exist
        assert not cache_exists(key)

    def test_clear_all_cache(self, sample_function):
        """Test clearing all cache entries."""
        cached_func1 = cache(sample_function)

        def multiply_func(a, b):
            return a * b

        cached_func2 = cache(multiply_func)

        # Cache some results
        cached_func1(1, 2)
        cached_func2(3, 4)

        assert sample_function.call_count() == 1

        # Clear all cache
        clear_all_cache()

        # Verify cache is cleared by calling functions again
        cached_func1(1, 2)
        cached_func2(3, 4)
        assert sample_function.call_count() == 2


class TestCacheInternalFunctions:
    """Test internal cache functions."""

    def test_cache_function_call_with_debug(self, memory_backend):
        """Test internal cache function call with debug mode."""

        def test_func(a, b):
            return a + b

        with patch("cacheflow.core.logger") as mock_logger:
            result = _cache_function_call(
                func=test_func,
                args=(1, 2),
                kwargs={},
                cache_backend=memory_backend,
                name="test_func",
                name_fn=None,
                namespace=None,
                ttl=None,
                key_fn=None,
                debug=True,
            )

            assert result == 3
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_cache_function_call_async_with_debug(self, memory_backend):
        """Test internal async cache function call with debug mode."""

        async def test_func(a, b):
            return a + b

        with patch("cacheflow.core.logger") as mock_logger:
            result = await _cache_function_call_async(
                func=test_func,
                args=(1, 2),
                kwargs={},
                cache_backend=memory_backend,
                name="test_func",
                name_fn=None,
                namespace=None,
                ttl=None,
                key_fn=None,
                debug=True,
            )

            assert result == 3
            assert mock_logger.info.called

    def test_cache_function_call_with_skip(self, memory_backend):
        """Test internal cache function call with skip parameter."""
        call_count = 0

        def test_func(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # First call - should cache
        result1 = _cache_function_call(
            func=test_func,
            args=(1, 2),
            kwargs={},
            cache_backend=memory_backend,
            name="test_func",
            name_fn=None,
            namespace=None,
            ttl=None,
            key_fn=None,
            debug=False,
        )
        assert result1 == 3
        assert call_count == 1

        # Second call with skip - should not use cache
        result2 = _cache_function_call(
            func=test_func,
            args=(1, 2),
            kwargs={"_cacheflow_skip": True},
            cache_backend=memory_backend,
            name="test_func",
            name_fn=None,
            namespace=None,
            ttl=None,
            key_fn=None,
            debug=False,
        )
        assert result2 == 3
        assert call_count == 2

    def test_cache_function_call_with_custom_name_fn(self, memory_backend):
        """Test internal cache function call with custom name function."""

        def test_func(a, b):
            return a + b

        def custom_name_fn(args, kwargs):
            return f"custom_{args[0]}_{args[1]}"

        result = _cache_function_call(
            func=test_func,
            args=(1, 2),
            kwargs={},
            cache_backend=memory_backend,
            name=None,
            name_fn=custom_name_fn,
            namespace=None,
            ttl=None,
            key_fn=None,
            debug=False,
        )

        assert result == 3

        # Verify custom name function was used
        keys = list(memory_backend._cache.keys())
        assert any("custom_1_2:" in key for key in keys)

    def test_cache_function_call_hit_and_miss(self, memory_backend):
        """Test cache hit and miss scenarios."""

        def test_func(a, b):
            return a + b

        # First call - cache miss
        result1 = _cache_function_call(
            func=test_func,
            args=(1, 2),
            kwargs={},
            cache_backend=memory_backend,
            name="test_func",
            name_fn=None,
            namespace=None,
            ttl=None,
            key_fn=None,
            debug=False,
        )
        assert result1 == 3

        # Second call - cache hit
        result2 = _cache_function_call(
            func=test_func,
            args=(1, 2),
            kwargs={},
            cache_backend=memory_backend,
            name="test_func",
            name_fn=None,
            namespace=None,
            ttl=None,
            key_fn=None,
            debug=False,
        )
        assert result2 == 3

        # Verify cache was used
        assert len(memory_backend._cache) == 1
