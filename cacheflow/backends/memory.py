"""In-memory cache backend."""

import time
from typing import Any

from .base import CacheBackend


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""

    def __init__(self):
        """Initialize memory cache backend."""
        self._cache: dict[str, dict[str, Any]] = {}
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    def get(self, key: str) -> Any | None:
        """Get value from cache by key."""
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check if expired
        if entry.get("expires") and time.time() > entry["expires"]:
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1
        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        entry = {"value": value}
        if ttl is not None:
            entry["expires"] = time.time() + ttl

        self._cache[key] = entry
        self._stats["sets"] += 1

    def delete(self, key: str) -> bool:
        """Delete value from cache by key. Returns True if key existed."""
        if key in self._cache:
            del self._cache[key]
            self._stats["deletes"] += 1
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace. Returns number of deleted keys."""
        prefix = f"{namespace}:"
        keys_to_delete = [key for key in self._cache if key.startswith(prefix)]
        deleted_count = 0
        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1
        return deleted_count

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        # Check if expired
        if entry.get("expires") and time.time() > entry["expires"]:
            del self._cache[key]
            return False

        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        # Clean up expired entries for accurate size
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if entry.get("expires") and current_time > entry["expires"]
        ]
        for key in expired_keys:
            del self._cache[key]

        total_calls = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_calls if total_calls > 0 else 0

        return {
            "size": len(self._cache),
            "hit_rate": hit_rate,
            "total_calls": total_calls,
            **self._stats,
        }
