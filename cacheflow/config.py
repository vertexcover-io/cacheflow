"""Configuration system for CacheFlow."""

import os
from dataclasses import dataclass, field
from typing import Any

from .backends import CacheBackend, DiskCacheBackend, MemoryCacheBackend


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    enabled: bool = True
    backend: str = "disk"
    default_ttl: int | None = None
    cache_dir: str = "./.cache"
    namespace: str | None = None
    debug: bool = False

    # LLM-specific settings
    llm_provider: str = "openai"
    image_key_replacement: bool = True

    # Backend-specific settings
    redis_url: str | None = None
    s3_bucket: str | None = None

    # Internal
    _backend_instance: CacheBackend | None = field(default=None, init=False)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self.enabled = self._get_bool_env("CACHEFLOW_ENABLED", self.enabled)
        self.backend = os.getenv("CACHEFLOW_BACKEND", self.backend)
        self.default_ttl = self._get_int_env("CACHEFLOW_DEFAULT_TTL", self.default_ttl)
        self.cache_dir = os.getenv("CACHEFLOW_CACHE_DIR", self.cache_dir)
        self.namespace = os.getenv("CACHEFLOW_NAMESPACE", self.namespace)
        self.debug = self._get_bool_env("CACHEFLOW_DEBUG", self.debug)

        # LLM settings
        self.llm_provider = os.getenv("CACHEFLOW_LLM_PROVIDER", self.llm_provider)
        self.image_key_replacement = self._get_bool_env(
            "CACHEFLOW_IMAGE_KEY_REPLACEMENT", self.image_key_replacement
        )

        # Backend-specific settings
        self.redis_url = os.getenv("CACHEFLOW_REDIS_URL", self.redis_url)
        self.s3_bucket = os.getenv("CACHEFLOW_S3_BUCKET", self.s3_bucket)

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _get_int_env(self, key: str, default: int | None) -> int | None:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_backend(self) -> CacheBackend:
        """Get or create the cache backend instance."""
        if self._backend_instance is None:
            self._backend_instance = create_backend(
                self.backend, cache_dir=self.cache_dir
            )
        return self._backend_instance

    def reset_backend(self) -> None:
        """Reset the backend instance (useful for testing)."""
        self._backend_instance = None


def create_backend(backend: str, **kwargs: Any) -> CacheBackend:
    """Create a cache backend instance based on configuration."""
    if backend == "disk":
        return DiskCacheBackend(cache_dir=kwargs.get("cache_dir", "./.cache"))
    elif backend == "memory":
        return MemoryCacheBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Global configuration instance
_config = CacheConfig()


def configure(**kwargs: Any) -> None:
    """Update global cache configuration."""
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

    # Reset backend if backend-related settings changed
    if any(key in kwargs for key in ["backend", "cache_dir", "redis_url", "s3_bucket"]):
        _config.reset_backend()


def get_config() -> CacheConfig:
    """Get current global configuration."""
    return _config


def reset_config() -> None:
    """Reset configuration to defaults (useful for testing)."""
    global _config  # noqa: PLW0603
    _config = CacheConfig()
