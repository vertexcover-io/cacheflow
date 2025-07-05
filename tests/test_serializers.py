"""Unit tests for serialization utilities."""

import json
from datetime import datetime
from enum import Enum
from unittest.mock import patch

import pytest

from cacheflow.serializers import (
    cache_json_serializer,
    default_key_fn,
    json_serializer,
    normalize_cache_key,
)

# Test if pydantic is available
try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True

    class SampleModel(BaseModel):
        name: str
        age: int
        active: bool = True

    class NestedSampleModel(BaseModel):
        inner: SampleModel
        items: list[str]

except ImportError:
    BaseModel = None
    PYDANTIC_AVAILABLE = False
    SampleModel = None
    NestedSampleModel = None


class SampleEnum(Enum):
    """Test enum for serialization testing."""

    OPTION_A = "a"
    OPTION_B = "b"
    OPTION_C = 42


class TestJsonSerializer:
    """Test the custom JSON serializer."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_serialize_pydantic_model(self):
        """Test serializing Pydantic BaseModel objects."""
        model = SampleModel(name="John", age=30, active=True)
        result = json_serializer(model)

        expected = {"name": "John", "age": 30, "active": True}
        assert result == expected
        assert isinstance(result, dict)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_serialize_nested_pydantic_model(self):
        """Test serializing nested Pydantic models."""
        inner = SampleModel(name="Jane", age=25)
        nested = NestedSampleModel(inner=inner, items=["a", "b", "c"])
        result = json_serializer(nested)

        expected = {
            "inner": {"name": "Jane", "age": 25, "active": True},
            "items": ["a", "b", "c"],
        }
        assert result == expected

    def test_serialize_enum(self):
        """Test serializing Enum objects."""
        assert json_serializer(SampleEnum.OPTION_A) == "a"
        assert json_serializer(SampleEnum.OPTION_B) == "b"
        assert json_serializer(SampleEnum.OPTION_C) == 42

    def test_serialize_datetime(self):
        """Test serializing datetime objects."""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        result = json_serializer(dt)

        assert result == "2023-12-25T15:30:45"
        assert isinstance(result, str)

    def test_serialize_type_objects(self):
        """Test serializing type objects."""
        # Built-in types
        assert json_serializer(int) == "<Type:builtins.int>"
        assert json_serializer(str) == "<Type:builtins.str>"
        assert json_serializer(list) == "<Type:builtins.list>"

        # Custom class
        class CustomClass:
            pass

        result = json_serializer(CustomClass)
        assert result.startswith("<Type:")
        assert "CustomClass" in result

    def test_serialize_unsupported_type(self):
        """Test serializing unsupported types raises TypeError."""
        unsupported_objects = [
            object(),
            lambda x: x,
            complex(1, 2),
            {1, 2, 3},
            frozenset([1, 2, 3]),
        ]

        for obj in unsupported_objects:
            with pytest.raises(TypeError):
                json_serializer(obj)

    def test_serialize_with_json_dumps(self):
        """Test that serializer works with json.dumps."""
        data = {
            "enum": SampleEnum.OPTION_A,
            "datetime": datetime(2023, 1, 1),
            "type": int,
        }

        # Should not raise exception
        json_str = json.dumps(data, default=json_serializer)

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["enum"] == "a"
        assert parsed["datetime"] == "2023-01-01T00:00:00"
        assert parsed["type"] == "<Type:builtins.int>"


class TestCacheJsonSerializer:
    """Test the cache-specific JSON serializer."""

    def test_serialize_bytes_returns_none(self):
        """Test that bytes objects return None."""
        assert cache_json_serializer(b"hello") is None
        assert cache_json_serializer(bytes([1, 2, 3])) is None

    def test_serialize_delegates_to_json_serializer(self):
        """Test that other types are delegated to json_serializer."""
        # Test enum
        assert cache_json_serializer(SampleEnum.OPTION_A) == "a"

        # Test datetime
        dt = datetime(2023, 1, 1)
        assert cache_json_serializer(dt) == "2023-01-01T00:00:00"

        # Test type
        assert cache_json_serializer(int) == "<Type:builtins.int>"

    def test_serialize_handles_serialization_errors(self):
        """Test that serialization errors are handled gracefully."""
        # Object that can't be serialized
        unsupported_obj = object()

        with patch("cacheflow.serializers.logging.warning") as mock_warning:
            result = cache_json_serializer(unsupported_obj)

            # Should return string representation
            assert isinstance(result, str)
            assert "object" in result.lower()

            # Should log warning
            mock_warning.assert_called_once()

    def test_serialize_handles_recursion_error(self):
        """Test that RecursionError is handled gracefully."""
        # Create circular reference
        circular_dict = {}
        circular_dict["self"] = circular_dict

        with patch("cacheflow.serializers.logging.warning") as mock_warning:
            result = cache_json_serializer(circular_dict)

            # Should return string representation
            assert isinstance(result, str)

            # Should log warning
            mock_warning.assert_called_once()

    def test_serialize_with_json_dumps(self):
        """Test that cache serializer works with json.dumps."""
        data = {
            "string": "hello",
            "bytes": b"binary_data",
            "enum": SampleEnum.OPTION_B,
            "unsupported": object(),
        }

        # Should not raise exception
        json_str = json.dumps(data, default=cache_json_serializer)

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["string"] == "hello"
        assert parsed["bytes"] is None
        assert parsed["enum"] == "b"
        assert isinstance(parsed["unsupported"], str)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_serialize_pydantic_model(self):
        """Test serializing Pydantic models through cache serializer."""
        model = SampleModel(name="Alice", age=28)
        result = cache_json_serializer(model)

        expected = {"name": "Alice", "age": 28, "active": True}
        assert result == expected


class TestNormalizeCacheKey:
    """Test cache key normalization."""

    def test_normalize_basic_string(self):
        """Test normalizing basic string."""
        assert normalize_cache_key("SimpleKey") == "simplekey"
        assert normalize_cache_key("UPPERCASE") == "uppercase"
        assert normalize_cache_key("MixedCase") == "mixedcase"

    def test_normalize_with_whitespace(self):
        """Test normalizing strings with whitespace."""
        assert normalize_cache_key("  spaced  ") == "spaced"
        assert normalize_cache_key("key with spaces") == "key_with_spaces"
        assert normalize_cache_key("\tkey\twith\ttabs\t") == "key_with_tabs"
        assert normalize_cache_key("\nkey\nwith\nnewlines\n") == "key_with_newlines"

    def test_normalize_with_special_characters(self):
        """Test normalizing strings with special characters."""
        assert normalize_cache_key("key:with:colons") == "key_with_colons"
        assert normalize_cache_key("key/with/slashes") == "key_with_slashes"
        assert normalize_cache_key("key\\with\\backslashes") == "key_with_backslashes"
        assert normalize_cache_key("key:with/mixed\\chars") == "key_with_mixed_chars"

    def test_normalize_complex_key(self):
        """Test normalizing complex cache key."""
        complex_key = " Complex:Key/With\\Many SPECIAL chars "
        expected = "complex_key_with_many_special_chars"
        assert normalize_cache_key(complex_key) == expected

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        assert normalize_cache_key("") == ""
        assert normalize_cache_key("   ") == ""

    def test_normalize_already_normalized(self):
        """Test normalizing already normalized keys."""
        normalized = "already_normalized_key"
        assert normalize_cache_key(normalized) == normalized

    def test_normalize_numeric_strings(self):
        """Test normalizing numeric strings."""
        assert normalize_cache_key("123") == "123"
        assert normalize_cache_key("3.14") == "3_14"  # dot becomes underscore
        assert normalize_cache_key("1,000") == "1,000"  # comma unchanged


class TestDefaultKeyFn:
    """Test default key generation function."""

    def test_default_key_fn_simple_args(self):
        """Test key generation with simple arguments."""
        args = (1, 2, 3)
        kwargs = {"key": "value"}

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_default_key_fn_different_args(self):
        """Test that different arguments produce different keys."""
        key1 = default_key_fn((1, 2), {})
        key2 = default_key_fn((1, 3), {})
        key3 = default_key_fn((), {"a": 1})
        key4 = default_key_fn((), {"a": 2})

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key1 != key4
        assert key2 != key3
        assert key2 != key4
        assert key3 != key4

    def test_default_key_fn_empty_args(self):
        """Test key generation with empty arguments."""
        key = default_key_fn((), {})
        assert isinstance(key, str)
        assert len(key) == 32

    def test_default_key_fn_complex_data(self):
        """Test key generation with complex data structures."""
        args = ([1, 2, 3], {"nested": True})
        kwargs = {
            "list": [4, 5, 6],
            "dict": {"inner": {"deep": "value"}},
            "tuple": (7, 8, 9),
        }

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2

    def test_default_key_fn_with_datetime(self):
        """Test key generation with datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        args = (dt,)
        kwargs = {"timestamp": dt}

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2

    def test_default_key_fn_with_enum(self):
        """Test key generation with enum objects."""
        args = (SampleEnum.OPTION_A,)
        kwargs = {"option": SampleEnum.OPTION_B}

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_default_key_fn_with_pydantic(self):
        """Test key generation with Pydantic models."""
        model = SampleModel(name="Test", age=25)
        args = (model,)
        kwargs = {"model": model}

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2

    def test_default_key_fn_with_bytes(self):
        """Test key generation with bytes objects."""
        args = (b"binary_data",)
        kwargs = {"data": b"more_binary"}

        # Should not raise exception
        key = default_key_fn(args, kwargs)
        assert isinstance(key, str)
        assert len(key) == 32

    def test_default_key_fn_deterministic(self):
        """Test that key generation is deterministic."""
        args = (1, "test", [1, 2, 3])
        kwargs = {"key": "value", "number": 42}

        # Generate keys multiple times
        keys = [default_key_fn(args, kwargs) for _ in range(10)]

        # All keys should be identical
        first_key = keys[0]
        for key in keys:
            assert key == first_key

    def test_default_key_fn_order_independence(self):
        """Test that kwargs order doesn't affect key generation."""
        args = (1, 2)
        kwargs1 = {"a": 1, "b": 2, "c": 3}
        kwargs2 = {"c": 3, "a": 1, "b": 2}

        key1 = default_key_fn(args, kwargs1)
        key2 = default_key_fn(args, kwargs2)

        # Keys should be the same despite different order
        assert key1 == key2

    def test_default_key_fn_with_none_values(self):
        """Test key generation with None values."""
        args = (None, 1, None)
        kwargs = {"key": None, "value": "test", "empty": None}

        key1 = default_key_fn(args, kwargs)
        key2 = default_key_fn(args, kwargs)

        assert key1 == key2

    def test_default_key_fn_json_serialization_errors(self):
        """Test key generation when JSON serialization encounters errors."""
        # Object that will cause serialization to fall back to string representation
        problematic_obj = object()
        args = (problematic_obj,)
        kwargs = {"obj": problematic_obj}

        # Should not raise exception, but use string fallback
        key = default_key_fn(args, kwargs)
        assert isinstance(key, str)
        assert len(key) == 32


class TestSerializationIntegration:
    """Test integration between different serialization components."""

    def test_end_to_end_serialization(self):
        """Test complete serialization pipeline."""
        # Complex data structure
        data = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "enum": SampleEnum.OPTION_A,
            "datetime": datetime(2023, 1, 1),
            "type": int,
            "bytes": b"binary",
        }

        # Serialize with cache serializer
        json_str = json.dumps(data, default=cache_json_serializer)

        # Should be valid JSON
        parsed = json.loads(json_str)

        # Verify serialization results
        assert parsed["string"] == "test"
        assert parsed["int"] == 42
        assert parsed["float"] == 3.14
        assert parsed["bool"] is True
        assert parsed["none"] is None
        assert parsed["list"] == [1, 2, 3]
        assert parsed["dict"] == {"nested": True}
        assert parsed["enum"] == "a"
        assert parsed["datetime"] == "2023-01-01T00:00:00"
        assert parsed["type"] == "<Type:builtins.int>"
        assert parsed["bytes"] is None

    def test_cache_key_generation_pipeline(self):
        """Test complete cache key generation pipeline."""
        # Create complex arguments
        args = (
            "function_name",
            42,
            datetime(2023, 1, 1),
            SampleEnum.OPTION_B,
        )
        kwargs = {
            "param1": "value1",
            "param2": [1, 2, 3],
            "param3": {"nested": {"deep": "value"}},
            "binary_data": b"binary",
        }

        # Generate key
        key = default_key_fn(args, kwargs)

        # Normalize key (though it's already normalized by hash)
        normalized_key = normalize_cache_key(key)

        # Should be valid cache key
        assert isinstance(key, str)
        assert isinstance(normalized_key, str)
        assert len(key) == 32
        assert key == normalized_key  # Hash is already normalized

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_integration_end_to_end(self):
        """Test end-to-end with Pydantic models."""
        model = SampleModel(name="Integration Test", age=30)
        nested = NestedSampleModel(inner=model, items=["a", "b", "c"])

        args = (nested,)
        kwargs = {"extra_model": model}

        # Should not raise exceptions
        key = default_key_fn(args, kwargs)
        assert isinstance(key, str)
        assert len(key) == 32

    def test_error_handling_integration(self):
        """Test error handling across serialization components."""
        # Create problematic data
        problematic_data = {
            "good": "data",
            "circular": {},
            "unsupported": object(),
            "bytes": b"binary",
        }
        problematic_data["circular"]["self"] = problematic_data["circular"]

        args = (problematic_data,)
        kwargs = {"more_problematic": object()}

        # Should handle errors gracefully
        with patch("cacheflow.serializers.logging.warning"):
            key = default_key_fn(args, kwargs)

        assert isinstance(key, str)
        assert len(key) == 32
