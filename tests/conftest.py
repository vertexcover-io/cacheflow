"""Pytest configuration and fixtures for CacheFlow tests."""

import shutil
import tempfile

import pytest

from cacheflow.backends.disk import DiskCacheBackend
from cacheflow.backends.memory import MemoryCacheBackend
from cacheflow.config import configure, reset_config


@pytest.fixture(autouse=True)
def reset_cache_config():
    """Reset cache configuration before each test."""
    reset_config()
    # Use memory backend for testing to avoid persistent cache
    configure(backend="memory")
    yield
    reset_config()


@pytest.fixture
def memory_backend():
    """Provide a fresh memory backend for testing."""
    return MemoryCacheBackend()


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary directory for disk cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def disk_backend(temp_cache_dir):
    """Provide a disk backend with temporary directory."""
    return DiskCacheBackend(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_function():
    """Provide a sample function for testing."""
    call_count = 0

    def _sample_function(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    def reset_count():
        nonlocal call_count
        call_count = 0

    # Add attributes to function object
    _sample_function.call_count = lambda: call_count  # type: ignore
    _sample_function.reset_count = reset_count  # type: ignore
    return _sample_function


@pytest.fixture
def sample_async_function():
    """Provide a sample async function for testing."""
    call_count = 0

    async def _sample_async_function(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    def reset_count():
        nonlocal call_count
        call_count = 0

    # Add attributes to function object
    _sample_async_function.call_count = lambda: call_count  # type: ignore
    _sample_async_function.reset_count = reset_count  # type: ignore
    return _sample_async_function


@pytest.fixture
def openai_messages_simple():
    """Provide simple OpenAI messages for testing."""
    return [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hi there!"},
    ]


@pytest.fixture
def openai_messages_with_images():
    """Provide OpenAI messages with images for testing."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "And this one?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                    },
                },
            ],
        },
    ]


@pytest.fixture
def anthropic_messages_with_images():
    """Provide Anthropic messages with images for testing."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
                    },
                },
            ],
        }
    ]


@pytest.fixture
def gemini_contents_with_images():
    """Provide Gemini contents with images for testing."""
    return [
        {
            "role": "user",
            "parts": [
                {"text": "What's in this image?"},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
                    }
                },
            ],
        }
    ]


@pytest.fixture
def image_keys():
    """Provide sample image keys for testing."""
    return ["img_key_1", "img_key_2", "img_key_3"]
