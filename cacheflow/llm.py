"""LLM-specific caching utilities."""

import hashlib
import json
import re
from collections.abc import Callable

from .core import cache
from .serializers import cache_json_serializer


def llm_cache(
    provider: str = "openai",
    ttl: int | None = None,
    namespace: str | None = None,
    key_fn: Callable | None = None,
    **kwargs,
):
    """
    LLM-specific caching with image content normalization.

    Args:
        provider: LLM provider ("openai", "anthropic", "gemini")
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        key_fn: Custom key generation function
        **kwargs: Additional arguments passed to cache decorator
    """
    # Use provider-specific key function if none provided
    if key_fn is None:
        key_fn = get_provider_key_fn(provider)

    # Set namespace based on provider if not provided
    if namespace is None:
        namespace = f"llm_{provider}"

    return cache(
        ttl=ttl,
        namespace=namespace,
        key_fn=key_fn,
        **kwargs,
    )


def get_provider_key_fn(provider: str) -> Callable:
    """Get provider-specific key generation function."""
    provider_map = {
        "openai": openai_key_fn,
        "anthropic": anthropic_key_fn,
        "gemini": gemini_key_fn,
    }
    return provider_map.get(provider, default_llm_key_fn)


def extract_image_keys(kwargs: dict) -> list:
    """Extract image keys from metadata."""
    metadata = kwargs.get("metadata", {})
    return metadata.get("image_keys", [])


def _generic_llm_key_fn(args: tuple, kwargs: dict, message_processor: Callable) -> str:
    """Generic LLM key generation with image handling."""
    # Extract image keys from metadata
    image_keys = extract_image_keys(kwargs)

    # Make a copy of kwargs to avoid modifying the original
    kwargs = dict(kwargs)

    # Process messages/content for image normalization
    kwargs = message_processor(kwargs, image_keys)

    # Remove metadata if present (not part of API call)
    kwargs = {k: v for k, v in kwargs.items() if k != "metadata"}

    json_key = {"args": args, "kwargs": kwargs}
    json_str = json.dumps(json_key, default=cache_json_serializer)
    return hashlib.md5(json_str.encode()).hexdigest()


def _process_openai_messages(kwargs: dict, image_keys: list) -> dict:
    """Process OpenAI messages for image normalization."""
    if "messages" in kwargs:
        kwargs["messages"] = _normalize_openai_messages(kwargs["messages"], image_keys)
    return kwargs


def _process_anthropic_messages(kwargs: dict, image_keys: list) -> dict:
    """Process Anthropic messages for image normalization."""
    if "messages" in kwargs:
        kwargs["messages"] = _normalize_anthropic_messages(kwargs["messages"], image_keys)
    return kwargs


def _process_gemini_contents(kwargs: dict, image_keys: list) -> dict:
    """Process Gemini contents for image normalization."""
    if "contents" in kwargs:
        kwargs["contents"] = _normalize_gemini_contents(kwargs["contents"], image_keys)
    return kwargs


def openai_key_fn(args: tuple, kwargs: dict) -> str:
    """OpenAI-specific key generation with image handling."""
    return _generic_llm_key_fn(args, kwargs, _process_openai_messages)


def anthropic_key_fn(args: tuple, kwargs: dict) -> str:
    """Anthropic-specific key generation."""
    return _generic_llm_key_fn(args, kwargs, _process_anthropic_messages)


def gemini_key_fn(args: tuple, kwargs: dict) -> str:
    """Gemini-specific key generation."""
    return _generic_llm_key_fn(args, kwargs, _process_gemini_contents)


def default_llm_key_fn(args: tuple, kwargs: dict) -> str:
    """Default LLM key generation without provider-specific handling."""
    return _generic_llm_key_fn(args, kwargs, lambda k, _: k)


def _normalize_openai_messages(messages: list[dict], image_keys: list) -> list[dict]:
    """Normalize OpenAI messages by replacing image URLs with placeholders."""
    normalized = []
    image_index = 0

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message["content"]

            # Handle string content
            if isinstance(content, str):
                normalized.append(message)
            # Handle array content (vision API)
            elif isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        # Replace with image key if available, otherwise use placeholder
                        if image_index < len(image_keys):
                            replacement_url = f"image_key:{image_keys[image_index]}"
                        else:
                            replacement_url = _get_image_placeholder(url)

                        image_index += 1
                        normalized_item = {
                            "type": "image_url",
                            "image_url": {"url": replacement_url},
                        }
                        normalized_content.append(normalized_item)
                    else:
                        normalized_content.append(item)

                normalized_message = dict(message)
                normalized_message["content"] = normalized_content
                normalized.append(normalized_message)
            else:
                normalized.append(message)
        else:
            normalized.append(message)

    return normalized


def _normalize_anthropic_messages(messages: list[dict], image_keys: list) -> list[dict]:
    """Normalize Anthropic messages by replacing image content with placeholders."""
    normalized = []
    image_index = 0

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message["content"]

            # Handle string content
            if isinstance(content, str):
                normalized.append(message)
            # Handle array content
            elif isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_data = item["source"]["data"]
                        # Replace with image key if available, otherwise use placeholder
                        if image_index < len(image_keys):
                            replacement_data = f"image_key:{image_keys[image_index]}"
                        else:
                            replacement_data = _get_image_placeholder(image_data)

                        image_index += 1
                        normalized_item = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": item["source"]["media_type"],
                                "data": replacement_data,
                            },
                        }
                        normalized_content.append(normalized_item)
                    else:
                        normalized_content.append(item)

                normalized_message = dict(message)
                normalized_message["content"] = normalized_content
                normalized.append(normalized_message)
            else:
                normalized.append(message)
        else:
            normalized.append(message)

    return normalized


def _normalize_gemini_contents(contents: list[dict], image_keys: list) -> list[dict]:
    """Normalize Gemini contents by replacing image data with placeholders."""
    normalized = []
    image_index = 0

    for content in contents:
        if isinstance(content, dict) and "parts" in content:
            parts = content["parts"]
            normalized_parts = []

            for part in parts:
                if isinstance(part, dict) and "inline_data" in part:
                    image_data = part["inline_data"]["data"]
                    # Replace with image key if available, otherwise use placeholder
                    if image_index < len(image_keys):
                        replacement_data = f"image_key:{image_keys[image_index]}"
                    else:
                        replacement_data = _get_image_placeholder(image_data)

                    image_index += 1
                    normalized_part = {
                        "inline_data": {
                            "mime_type": part["inline_data"]["mime_type"],
                            "data": replacement_data,
                        }
                    }
                    normalized_parts.append(normalized_part)
                else:
                    normalized_parts.append(part)

            normalized_content = dict(content)
            normalized_content["parts"] = normalized_parts
            normalized.append(normalized_content)
        else:
            normalized.append(content)

    return normalized


def _get_image_placeholder(image_data: str) -> str:
    """
    Generate a placeholder for image data based on metadata or content hash.

    This function looks for image keys in metadata and uses them as placeholders.
    If no metadata is available, it creates a hash of the image data.
    """
    # Check if this looks like a URL
    if image_data.startswith(("http://", "https://")):
        # Extract domain and path for URL-based placeholder
        url_match = re.match(r"https?://([^/]+)(/.*)?", image_data)
        if url_match:
            domain = url_match.group(1)
            path = url_match.group(2) or ""
            return f"image_url:{domain}{path}"

    # Check if this looks like base64 data
    if len(image_data) > 100:  # Likely base64 encoded image
        # Create a hash of the first and last 50 characters
        # This preserves uniqueness while being cacheable
        start = image_data[:50]
        end = image_data[-50:]
        content_hash = hashlib.md5(f"{start}{end}".encode()).hexdigest()[:8]
        return f"image_data:{content_hash}"

    # For short strings, use as-is but prefix to mark as placeholder
    return f"image_placeholder:{image_data}"


def replace_image_urls_with_keys(
    messages: list[dict], image_keys: list[str]
) -> list[dict]:
    """
    Replace image URLs in messages with keys from metadata.

    This is useful when you have pre-computed image keys and want to use them
    for consistent caching across different calls.
    """
    if not image_keys:
        return messages

    normalized = []
    image_index = 0

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message["content"]

            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Replace with key if available
                        if image_index < len(image_keys):
                            normalized_item = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"image_key:{image_keys[image_index]}"
                                },
                            }
                            image_index += 1
                            normalized_content.append(normalized_item)
                        else:
                            normalized_content.append(item)
                    else:
                        normalized_content.append(item)

                normalized_message = dict(message)
                normalized_message["content"] = normalized_content
                normalized.append(normalized_message)
            else:
                normalized.append(message)
        else:
            normalized.append(message)

    return normalized
