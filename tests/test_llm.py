"""Unit tests for LLM-specific caching functionality."""

import time

from cacheflow.config import get_config
from cacheflow.llm import (
    _normalize_anthropic_messages,
    _normalize_gemini_contents,
    _normalize_openai_messages,
    _process_anthropic_messages,
    _process_gemini_contents,
    _process_openai_messages,
    anthropic_key_fn,
    default_llm_key_fn,
    extract_image_keys,
    gemini_key_fn,
    get_provider_key_fn,
    llm_cache,
    openai_key_fn,
    replace_image_urls_with_keys,
)


class TestLLMCacheDecorator:
    """Test the LLM cache decorator."""

    def test_llm_cache_basic_openai(self):
        """Test basic LLM cache with OpenAI provider."""
        call_count = 0

        def mock_openai_call(messages, model="gpt-3.5-turbo"):
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": "Hello!"}}]}

        cached_func = llm_cache(provider="openai")(mock_openai_call)

        # First call
        result1 = cached_func(messages=[{"role": "user", "content": "Hi"}])
        assert result1["choices"][0]["message"]["content"] == "Hello!"
        assert call_count == 1

        # Second call - should use cache
        result2 = cached_func(messages=[{"role": "user", "content": "Hi"}])
        assert result2["choices"][0]["message"]["content"] == "Hello!"
        assert call_count == 1

    def test_llm_cache_basic_anthropic(self):
        """Test basic LLM cache with Anthropic provider."""
        call_count = 0

        def mock_anthropic_call(messages, model="claude-3-sonnet-20240229"):
            nonlocal call_count
            call_count += 1
            return {"content": [{"text": "Hello!"}]}

        cached_func = llm_cache(provider="anthropic")(mock_anthropic_call)

        # First call
        result1 = cached_func(messages=[{"role": "user", "content": "Hi"}])
        assert result1["content"][0]["text"] == "Hello!"
        assert call_count == 1

        # Second call - should use cache
        result2 = cached_func(messages=[{"role": "user", "content": "Hi"}])
        assert result2["content"][0]["text"] == "Hello!"
        assert call_count == 1

    def test_llm_cache_basic_gemini(self):
        """Test basic LLM cache with Gemini provider."""
        call_count = 0

        def mock_gemini_call(contents, model="gemini-pro"):
            nonlocal call_count
            call_count += 1
            return {"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]}

        cached_func = llm_cache(provider="gemini")(mock_gemini_call)

        # First call
        result1 = cached_func(contents=[{"role": "user", "parts": [{"text": "Hi"}]}])
        assert result1["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"
        assert call_count == 1

        # Second call - should use cache
        result2 = cached_func(contents=[{"role": "user", "parts": [{"text": "Hi"}]}])
        assert result2["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"
        assert call_count == 1

    def test_llm_cache_with_custom_namespace(self):
        """Test LLM cache with custom namespace."""
        call_count = 0

        def mock_llm_call(messages):
            nonlocal call_count
            call_count += 1
            return {"response": "test"}

        cached_func = llm_cache(provider="openai", namespace="custom_ns")(mock_llm_call)

        result = cached_func(messages=[{"role": "user", "content": "test"}])
        assert result["response"] == "test"
        assert call_count == 1

        # Verify namespace is used

        backend = get_config().get_backend()
        if hasattr(backend, "_cache"):
            keys = list(backend._cache.keys())
            assert any("custom_ns:" in key for key in keys)

    def test_llm_cache_with_ttl(self):
        """Test LLM cache with TTL."""
        call_count = 0

        def mock_llm_call(messages):
            nonlocal call_count
            call_count += 1
            return {"response": "test"}

        cached_func = llm_cache(provider="openai", ttl=1)(mock_llm_call)

        # First call
        result1 = cached_func(messages=[{"role": "user", "content": "test"}])
        assert result1["response"] == "test"
        assert call_count == 1

        # Second call - should use cache
        result2 = cached_func(messages=[{"role": "user", "content": "test"}])
        assert result2["response"] == "test"
        assert call_count == 1

        time.sleep(1.1)

        # Third call - should execute function again
        result3 = cached_func(messages=[{"role": "user", "content": "test"}])
        assert result3["response"] == "test"
        assert call_count == 2

    def test_llm_cache_with_images_openai(self, openai_messages_with_images, image_keys):
        """Test LLM cache with image content for OpenAI."""
        call_count = 0

        def mock_openai_call(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"choices": [{"message": {"content": "I see an image"}}]}

        cached_func = llm_cache(provider="openai")(mock_openai_call)

        # First call with images
        result1 = cached_func(
            messages=openai_messages_with_images, metadata={"image_keys": image_keys}
        )
        assert result1["choices"][0]["message"]["content"] == "I see an image"
        assert call_count == 1

        # Second call with same images - should use cache
        result2 = cached_func(
            messages=openai_messages_with_images, metadata={"image_keys": image_keys}
        )
        assert result2["choices"][0]["message"]["content"] == "I see an image"
        assert call_count == 1

    def test_llm_cache_unknown_provider(self):
        """Test LLM cache with unknown provider falls back to default."""
        call_count = 0

        def mock_llm_call(messages):
            nonlocal call_count
            call_count += 1
            return {"response": "test"}

        cached_func = llm_cache(provider="unknown_provider")(mock_llm_call)

        result = cached_func(messages=[{"role": "user", "content": "test"}])
        assert result["response"] == "test"
        assert call_count == 1


class TestProviderKeyFunctions:
    """Test provider-specific key generation functions."""

    def test_get_provider_key_fn_openai(self):
        """Test getting OpenAI key function."""
        key_fn = get_provider_key_fn("openai")
        assert key_fn == openai_key_fn

    def test_get_provider_key_fn_anthropic(self):
        """Test getting Anthropic key function."""
        key_fn = get_provider_key_fn("anthropic")
        assert key_fn == anthropic_key_fn

    def test_get_provider_key_fn_gemini(self):
        """Test getting Gemini key function."""
        key_fn = get_provider_key_fn("gemini")
        assert key_fn == gemini_key_fn

    def test_get_provider_key_fn_unknown(self):
        """Test getting unknown provider falls back to default."""
        key_fn = get_provider_key_fn("unknown")
        assert key_fn == default_llm_key_fn

    def test_openai_key_fn_basic(self, openai_messages_simple):
        """Test OpenAI key function with simple messages."""
        args = ()
        kwargs = {"messages": openai_messages_simple}

        key1 = openai_key_fn(args, kwargs)
        key2 = openai_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_openai_key_fn_with_images(self, openai_messages_with_images, image_keys):
        """Test OpenAI key function with image messages."""
        args = ()
        kwargs = {
            "messages": openai_messages_with_images,
            "metadata": {"image_keys": image_keys},
        }

        key1 = openai_key_fn(args, kwargs)
        key2 = openai_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32

    def test_anthropic_key_fn_with_images(
        self, anthropic_messages_with_images, image_keys
    ):
        """Test Anthropic key function with image messages."""
        args = ()
        kwargs = {
            "messages": anthropic_messages_with_images,
            "metadata": {"image_keys": image_keys},
        }

        key1 = anthropic_key_fn(args, kwargs)
        key2 = anthropic_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32

    def test_gemini_key_fn_with_images(self, gemini_contents_with_images, image_keys):
        """Test Gemini key function with image contents."""
        args = ()
        kwargs = {
            "contents": gemini_contents_with_images,
            "metadata": {"image_keys": image_keys},
        }

        key1 = gemini_key_fn(args, kwargs)
        key2 = gemini_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32

    def test_default_llm_key_fn(self):
        """Test default LLM key function."""
        args = ()
        kwargs = {"messages": [{"role": "user", "content": "test"}]}

        key1 = default_llm_key_fn(args, kwargs)
        key2 = default_llm_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32


class TestImageProcessing:
    """Test image processing and normalization functions."""

    def test_extract_image_keys_with_metadata(self, image_keys):
        """Test extracting image keys from metadata."""
        kwargs = {"metadata": {"image_keys": image_keys}}
        extracted = extract_image_keys(kwargs)
        assert extracted == image_keys

    def test_extract_image_keys_without_metadata(self):
        """Test extracting image keys without metadata."""
        kwargs = {}
        extracted = extract_image_keys(kwargs)
        assert extracted == []

    def test_extract_image_keys_empty_metadata(self):
        """Test extracting image keys with empty metadata."""
        kwargs = {"metadata": {}}
        extracted = extract_image_keys(kwargs)
        assert extracted == []

    def test_normalize_openai_messages_simple(self, openai_messages_simple):
        """Test normalizing simple OpenAI messages."""
        result = _normalize_openai_messages(openai_messages_simple, [])
        assert result == openai_messages_simple

    def test_normalize_openai_messages_with_images(
        self, openai_messages_with_images, image_keys
    ):
        """Test normalizing OpenAI messages with images."""
        result = _normalize_openai_messages(openai_messages_with_images, image_keys)

        # Check that image URLs were replaced
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        assert (
                            url.startswith("image_key:")
                            or url.startswith("image_url:")
                            or url.startswith("image_data:")
                        )

    def test_normalize_openai_messages_without_image_keys(
        self, openai_messages_with_images
    ):
        """Test normalizing OpenAI messages without image keys."""
        result = _normalize_openai_messages(openai_messages_with_images, [])

        # Check that placeholders were used
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        assert not url.startswith("image_key:")

    def test_normalize_anthropic_messages_with_images(
        self, anthropic_messages_with_images, image_keys
    ):
        """Test normalizing Anthropic messages with images."""
        result = _normalize_anthropic_messages(anthropic_messages_with_images, image_keys)

        # Check that image data was replaced
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image":
                        data = item["source"]["data"]
                        assert data.startswith("image_key:") or data.startswith(
                            "image_data:"
                        )

    def test_normalize_gemini_contents_with_images(
        self, gemini_contents_with_images, image_keys
    ):
        """Test normalizing Gemini contents with images."""
        result = _normalize_gemini_contents(gemini_contents_with_images, image_keys)

        # Check that image data was replaced
        for content in result:
            if "parts" in content:
                for part in content["parts"]:
                    if "inline_data" in part:
                        data = part["inline_data"]["data"]
                        assert data.startswith("image_key:") or data.startswith(
                            "image_data:"
                        )

    def test_replace_image_urls_with_keys(self, image_keys):
        """Test replacing image URLs with keys."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image1.jpg"},
                    },
                ],
            }
        ]

        result = replace_image_urls_with_keys(messages, image_keys)

        # Check that URL was replaced with key
        image_item = result[0]["content"][1]
        assert image_item["image_url"]["url"] == f"image_key:{image_keys[0]}"

    def test_replace_image_urls_with_keys_empty_keys(self):
        """Test replacing image URLs with empty keys."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image1.jpg"},
                    }
                ],
            }
        ]

        result = replace_image_urls_with_keys(messages, [])

        # Should return original messages
        assert result == messages


class TestMessageProcessors:
    """Test message processor functions."""

    def test_process_openai_messages_basic(self, openai_messages_simple):
        """Test processing OpenAI messages without images."""
        kwargs = {"messages": openai_messages_simple}
        result = _process_openai_messages(kwargs, [])

        assert "messages" in result
        assert result["messages"] == openai_messages_simple

    def test_process_openai_messages_with_images(
        self, openai_messages_with_images, image_keys
    ):
        """Test processing OpenAI messages with images."""
        kwargs = {"messages": openai_messages_with_images}
        result = _process_openai_messages(kwargs, image_keys)

        assert "messages" in result
        # Messages should be normalized
        assert result["messages"] != openai_messages_with_images

    def test_process_openai_messages_without_messages(self):
        """Test processing OpenAI kwargs without messages."""
        kwargs = {"model": "gpt-3.5-turbo"}
        result = _process_openai_messages(kwargs, [])

        assert result == kwargs

    def test_process_anthropic_messages_with_images(
        self, anthropic_messages_with_images, image_keys
    ):
        """Test processing Anthropic messages with images."""
        kwargs = {"messages": anthropic_messages_with_images}
        result = _process_anthropic_messages(kwargs, image_keys)

        assert "messages" in result
        # Messages should be normalized
        assert result["messages"] != anthropic_messages_with_images

    def test_process_gemini_contents_with_images(
        self, gemini_contents_with_images, image_keys
    ):
        """Test processing Gemini contents with images."""
        kwargs = {"contents": gemini_contents_with_images}
        result = _process_gemini_contents(kwargs, image_keys)

        assert "contents" in result
        # Contents should be normalized
        assert result["contents"] != gemini_contents_with_images


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_normalize_messages_with_malformed_content(self):
        """Test normalizing messages with malformed content."""
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": 123},
            {"role": "user"},
            {"not_a_message": "invalid"},
        ]

        result = _normalize_openai_messages(messages, [])

        # Should handle malformed messages gracefully
        assert len(result) == len(messages)

    def test_normalize_messages_with_empty_list(self):
        """Test normalizing empty message list."""
        result = _normalize_openai_messages([], [])
        assert result == []

    def test_key_generation_with_complex_data(self):
        """Test key generation with complex nested data."""
        args = ()
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": {"nested": {"data": [1, 2, 3], "more": {"complex": True}}},
                }
            ]
        }

        key1 = openai_key_fn(args, kwargs)
        key2 = openai_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)

    def test_key_generation_removes_metadata(self):
        """Test that metadata is removed from key generation."""
        args = ()
        kwargs1 = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"image_keys": ["key1", "key2"]},
        }
        kwargs2 = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"image_keys": ["key3", "key4"]},
        }

        key1 = openai_key_fn(args, kwargs1)
        key2 = openai_key_fn(args, kwargs2)

        # Keys should be the same since metadata is removed
        assert key1 == key2

    def test_image_processing_with_insufficient_keys(self, openai_messages_with_images):
        """Test image processing when there are insufficient image keys."""
        # Only provide one key but there are two images
        image_keys = ["key1"]

        result = _normalize_openai_messages(openai_messages_with_images, image_keys)

        # Should handle gracefully and use placeholders for missing keys
        image_count = 0
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        image_count += 1

        assert image_count == 2  # Both images should be processed

    def test_image_processing_with_excess_keys(self, openai_messages_with_images):
        """Test image processing when there are excess image keys."""
        # Provide more keys than images
        image_keys = ["key1", "key2", "key3", "key4"]

        result = _normalize_openai_messages(openai_messages_with_images, image_keys)

        # Should use only the needed keys
        used_keys = []
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("image_key:"):
                            used_keys.append(url.split(":", 1)[1])

        assert len(used_keys) <= len(image_keys)
