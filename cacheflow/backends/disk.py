"""Disk-based cache backend using diskcache."""

from pathlib import Path
from typing import Any

import diskcache

from .base import CacheBackend


class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend using diskcache."""

    def __init__(self, cache_dir: str = "./.cache"):
        """Initialize disk cache backend."""
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir)

    def get(self, key: str) -> Any | None:
        """Get value from cache by key."""
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        if ttl is not None:
            self._cache.set(key, value, expire=ttl)
        else:
            self._cache.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete value from cache by key. Returns True if key existed."""
        return self._cache.delete(key)

    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace. Returns number of deleted keys."""
        prefix = f"{namespace}:"
        keys_to_delete = [key for key in self._cache if key.startswith(prefix)]
        deleted_count = 0
        for key in keys_to_delete:
            if self._cache.delete(key):
                deleted_count += 1
        return deleted_count

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "volume": self._cache.volume(),
            "cache_dir": self.cache_dir,
        }

    def __del__(self):
        """Close the cache when the object is destroyed."""
        if hasattr(self, "_cache"):
            self._cache.close()
