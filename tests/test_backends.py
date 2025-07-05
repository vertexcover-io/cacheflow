"""Unit tests for cache backends."""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cacheflow.backends.base import CacheBackend
from cacheflow.backends.disk import DiskCacheBackend
from cacheflow.backends.memory import MemoryCacheBackend


class TestCacheBackendInterface:
    """Test the abstract CacheBackend interface."""

    def test_cache_backend_is_abstract(self):
        """Test that CacheBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CacheBackend()

    def test_cache_backend_abstract_methods(self):
        """Test that all required methods are abstract."""
        abstract_methods = CacheBackend.__abstractmethods__
        expected_methods = {
            "get",
            "set",
            "delete",
            "clear_namespace",
            "exists",
            "clear",
            "get_stats",
        }
        assert abstract_methods == expected_methods


class TestMemoryCacheBackend:
    """Test the memory cache backend."""

    @pytest.fixture
    def cache(self):
        """Provide a fresh memory cache backend."""
        return MemoryCacheBackend()

    def test_memory_cache_initialization(self, cache):
        """Test memory cache backend initialization."""
        assert isinstance(cache._cache, dict)
        assert cache._cache == {}
        assert cache._stats["hits"] == 0
        assert cache._stats["misses"] == 0
        assert cache._stats["sets"] == 0
        assert cache._stats["deletes"] == 0

    def test_memory_cache_set_and_get(self, cache):
        """Test basic set and get operations."""
        # Set a value
        cache.set("test_key", "test_value")
        assert cache._stats["sets"] == 1

        # Get the value
        result = cache.get("test_key")
        assert result == "test_value"
        assert cache._stats["hits"] == 1
        assert cache._stats["misses"] == 0

    def test_memory_cache_get_nonexistent_key(self, cache):
        """Test getting a non-existent key."""
        result = cache.get("nonexistent_key")
        assert result is None
        assert cache._stats["misses"] == 1
        assert cache._stats["hits"] == 0

    def test_memory_cache_set_with_ttl(self, cache):
        """Test setting value with TTL."""
        # Set with short TTL
        cache.set("ttl_key", "ttl_value", ttl=1)

        # Should be available immediately
        result = cache.get("ttl_key")
        assert result == "ttl_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired and return None
        result = cache.get("ttl_key")
        assert result is None
        assert cache._stats["misses"] == 1

    def test_memory_cache_delete_existing_key(self, cache):
        """Test deleting an existing key."""
        cache.set("delete_me", "value")

        deleted = cache.delete("delete_me")
        assert deleted is True
        assert cache._stats["deletes"] == 1

        # Verify key is gone
        result = cache.get("delete_me")
        assert result is None

    def test_memory_cache_delete_nonexistent_key(self, cache):
        """Test deleting a non-existent key."""
        deleted = cache.delete("nonexistent")
        assert deleted is False
        assert cache._stats["deletes"] == 0

    def test_memory_cache_exists(self, cache):
        """Test checking if key exists."""
        # Non-existent key
        assert cache.exists("nonexistent") is False

        # Set a key
        cache.set("existing_key", "value")
        assert cache.exists("existing_key") is True

        # Delete the key
        cache.delete("existing_key")
        assert cache.exists("existing_key") is False

    def test_memory_cache_exists_with_expired_key(self, cache):
        """Test exists with expired key."""
        cache.set("expired_key", "value", ttl=1)

        # Should exist initially
        assert cache.exists("expired_key") is True

        # Wait for expiration
        time.sleep(1.1)

        # Should not exist after expiration
        assert cache.exists("expired_key") is False

    def test_memory_cache_clear_namespace(self, cache):
        """Test clearing namespace."""
        # Set keys in different namespaces
        cache.set("ns1:key1", "value1")
        cache.set("ns1:key2", "value2")
        cache.set("ns2:key1", "value3")
        cache.set("no_namespace", "value4")

        # Clear ns1 namespace
        deleted_count = cache.clear_namespace("ns1")
        assert deleted_count == 2

        # Verify only ns1 keys were deleted
        assert cache.get("ns1:key1") is None
        assert cache.get("ns1:key2") is None
        assert cache.get("ns2:key1") == "value3"
        assert cache.get("no_namespace") == "value4"

    def test_memory_cache_clear(self, cache):
        """Test clearing all cache entries."""
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Clear all
        cache.clear()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["deletes"] == 0

        assert cache.exists("key1") is False
        assert cache.exists("key2") is False
        assert cache.exists("key3") is False

    def test_memory_cache_get_stats(self, cache):
        """Test getting cache statistics."""
        # Make some operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 2
        assert stats["deletes"] == 0
        assert stats["total_calls"] == 3
        assert stats["hit_rate"] == 2 / 3

    def test_memory_cache_get_stats_empty(self, cache):
        """Test getting stats from empty cache."""
        stats = cache.get_stats()

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["deletes"] == 0
        assert stats["total_calls"] == 0
        assert stats["hit_rate"] == 0

    def test_memory_cache_get_stats_cleans_expired(self, cache):
        """Test that get_stats cleans up expired entries."""
        # Set some values with short TTL
        cache.set("temp1", "value1", ttl=1)
        cache.set("temp2", "value2", ttl=1)
        cache.set("permanent", "value3")

        # Initial size
        stats = cache.get_stats()
        assert stats["size"] == 3

        # Wait for expiration
        time.sleep(1.1)

        # Get stats should clean up expired entries
        stats = cache.get_stats()
        assert stats["size"] == 1  # Only permanent key remains

    def test_memory_cache_with_complex_data(self, cache):
        """Test caching complex data structures."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "tuple": (4, 5, 6),
            "none": None,
            "bool": True,
        }

        cache.set("complex", complex_data)
        result = cache.get("complex")

        assert result == complex_data
        assert isinstance(result, dict)

    def test_memory_cache_key_collision_resistance(self, cache):
        """Test that similar keys don't collide."""
        cache.set("key", "value1")
        cache.set("key2", "value2")
        cache.set("key:", "value3")
        cache.set(":key", "value4")

        assert cache.get("key") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key:") == "value3"
        assert cache.get(":key") == "value4"


class TestDiskCacheBackend:
    """Test the disk cache backend."""

    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache(self, temp_dir):
        """Provide a fresh disk cache backend."""
        return DiskCacheBackend(cache_dir=temp_dir)

    def test_disk_cache_initialization(self, temp_dir):
        """Test disk cache backend initialization."""
        cache = DiskCacheBackend(cache_dir=temp_dir)

        # Verify directory was created
        assert Path(temp_dir).exists()
        assert Path(temp_dir).is_dir()

        # Verify cache object
        assert hasattr(cache, "_cache")
        assert cache.cache_dir == temp_dir

    def test_disk_cache_initialization_creates_directory(self):
        """Test that initialization creates non-existent directory."""
        temp_dir = tempfile.mkdtemp()
        shutil.rmtree(temp_dir)  # Remove it

        assert not Path(temp_dir).exists()

        DiskCacheBackend(cache_dir=temp_dir)

        # Should create the directory
        assert Path(temp_dir).exists()
        assert Path(temp_dir).is_dir()

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_disk_cache_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"

    def test_disk_cache_get_nonexistent_key(self, cache):
        """Test getting a non-existent key."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_disk_cache_set_with_ttl(self, cache):
        """Test setting value with TTL."""
        cache.set("ttl_key", "ttl_value", ttl=1)

        # Should be available immediately
        result = cache.get("ttl_key")
        assert result == "ttl_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        result = cache.get("ttl_key")
        assert result is None

    def test_disk_cache_delete_existing_key(self, cache):
        """Test deleting an existing key."""
        cache.set("delete_me", "value")

        deleted = cache.delete("delete_me")
        assert deleted is True

        # Verify key is gone
        result = cache.get("delete_me")
        assert result is None

    def test_disk_cache_delete_nonexistent_key(self, cache):
        """Test deleting a non-existent key."""
        deleted = cache.delete("nonexistent")
        assert deleted is False

    def test_disk_cache_exists(self, cache):
        """Test checking if key exists."""
        assert cache.exists("nonexistent") is False

        cache.set("existing_key", "value")
        assert cache.exists("existing_key") is True

        cache.delete("existing_key")
        assert cache.exists("existing_key") is False

    def test_disk_cache_clear_namespace(self, cache):
        """Test clearing namespace."""
        # Set keys in different namespaces
        cache.set("ns1:key1", "value1")
        cache.set("ns1:key2", "value2")
        cache.set("ns2:key1", "value3")
        cache.set("no_namespace", "value4")

        # Clear ns1 namespace
        deleted_count = cache.clear_namespace("ns1")
        assert deleted_count == 2

        # Verify only ns1 keys were deleted
        assert cache.get("ns1:key1") is None
        assert cache.get("ns1:key2") is None
        assert cache.get("ns2:key1") == "value3"
        assert cache.get("no_namespace") == "value4"

    def test_disk_cache_clear(self, cache):
        """Test clearing all cache entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.clear()

        # Verify all keys are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_disk_cache_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "volume" in stats
        assert "cache_dir" in stats
        assert stats["cache_dir"] == cache.cache_dir
        assert stats["size"] >= 2

    def test_disk_cache_persistence(self, temp_dir):
        """Test that data persists across cache instances."""
        # Create first cache instance
        cache1 = DiskCacheBackend(cache_dir=temp_dir)
        cache1.set("persistent_key", "persistent_value")

        # Create second cache instance with same directory
        cache2 = DiskCacheBackend(cache_dir=temp_dir)
        result = cache2.get("persistent_key")

        assert result == "persistent_value"

    def test_disk_cache_with_complex_data(self, cache):
        """Test caching complex data structures."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "tuple": (4, 5, 6),
            "none": None,
            "bool": True,
        }

        cache.set("complex", complex_data)
        result = cache.get("complex")

        assert result == complex_data

    def test_disk_cache_destructor(self, temp_dir):
        """Test that cache is properly closed on destruction."""
        cache = DiskCacheBackend(cache_dir=temp_dir)
        cache.set("test", "value")

        # Mock the close method to verify it's called
        with patch.object(cache._cache, "close") as mock_close:
            del cache
            mock_close.assert_called_once()

    def test_disk_cache_large_values(self, cache):
        """Test caching large values."""
        large_value = "x" * 10000  # 10KB string

        cache.set("large_key", large_value)
        result = cache.get("large_key")

        assert result == large_value
        assert len(result) == 10000

    def test_disk_cache_special_characters_in_keys(self, cache):
        """Test caching with special characters in keys."""
        special_keys = [
            "key with spaces",
            "key/with/slashes",
            "key:with:colons",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key@with@symbols",
        ]

        for i, key in enumerate(special_keys):
            cache.set(key, f"value_{i}")

        for i, key in enumerate(special_keys):
            result = cache.get(key)
            assert result == f"value_{i}"


class TestBackendComparison:
    """Test comparing behavior between different backends."""

    @pytest.fixture
    def memory_cache(self):
        """Provide memory cache backend."""
        return MemoryCacheBackend()

    @pytest.fixture
    def disk_cache(self, temp_cache_dir):
        """Provide disk cache backend."""
        return DiskCacheBackend(cache_dir=temp_cache_dir)

    def test_backend_interface_consistency(self, memory_cache, disk_cache):
        """Test that both backends implement the same interface."""
        backends = [memory_cache, disk_cache]

        for backend in backends:
            # Test basic operations
            backend.set("test_key", "test_value")
            assert backend.get("test_key") == "test_value"
            assert backend.exists("test_key") is True
            assert backend.delete("test_key") is True
            assert backend.exists("test_key") is False

            # Test TTL
            backend.set("ttl_key", "ttl_value", ttl=3600)
            assert backend.get("ttl_key") == "ttl_value"

            # Test namespace operations
            backend.set("ns:key1", "value1")
            backend.set("ns:key2", "value2")
            deleted = backend.clear_namespace("ns")
            assert deleted == 2

            # Test stats
            stats = backend.get_stats()
            assert isinstance(stats, dict)
            assert "size" in stats

            # Test clear
            backend.set("clear_test", "value")
            backend.clear()
            assert backend.get("clear_test") is None


class TestBackendErrorHandling:
    """Test error handling in backends."""

    def test_memory_cache_with_none_key(self):
        """Test memory cache behavior with None key."""
        cache = MemoryCacheBackend()

        # Should handle None key gracefully
        cache.set(None, "value")
        result = cache.get(None)
        assert result == "value"

    def test_disk_cache_invalid_directory_permissions(self):
        """Test disk cache with invalid directory permissions."""
        # This test is platform-dependent and may not work on all systems
        if hasattr(Path, "chmod"):
            temp_dir = tempfile.mkdtemp()
            try:
                # Make directory read-only
                Path(temp_dir).chmod(0o444)

                # Should handle gracefully or raise appropriate exception
                with pytest.raises((PermissionError, OSError)):
                    DiskCacheBackend(cache_dir=f"{temp_dir}/readonly_subdir")

            finally:
                # Restore permissions and cleanup
                Path(temp_dir).chmod(0o755)
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_memory_cache_stats_with_no_operations(self):
        """Test memory cache stats when no operations have been performed."""
        cache = MemoryCacheBackend()
        stats = cache.get_stats()

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["deletes"] == 0
        assert stats["total_calls"] == 0
        assert stats["hit_rate"] == 0

    def test_backend_namespace_edge_cases(self):
        """Test namespace operations with edge cases."""
        cache = MemoryCacheBackend()

        # Empty namespace
        deleted = cache.clear_namespace("")
        assert deleted == 0

        # Namespace that matches key prefix
        cache.set("test", "value1")
        cache.set("test:key", "value2")
        deleted = cache.clear_namespace("test")
        assert deleted == 1  # Only "test:key" should be deleted
        assert cache.get("test") == "value1"  # "test" should remain

    def test_memory_cache_concurrent_operations(self):
        """Test memory cache with rapid concurrent-like operations."""
        cache = MemoryCacheBackend()

        # Rapid set/get operations
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
            result = cache.get(f"key_{i}")
            assert result == f"value_{i}"

        # Verify all keys exist
        for i in range(100):
            assert cache.exists(f"key_{i}")

        # Verify stats
        stats = cache.get_stats()
        assert stats["size"] == 100
        assert stats["sets"] == 100
        assert stats["hits"] == 100  # 100 from get operations + 100 from exists
