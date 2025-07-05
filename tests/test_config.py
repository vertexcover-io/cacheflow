import os
import tempfile
from unittest.mock import patch

import pytest

from cacheflow.backends.disk import DiskCacheBackend
from cacheflow.backends.memory import MemoryCacheBackend
from cacheflow.config import (
    CacheConfig,
    configure,
    create_backend,
    get_config,
    reset_config,
)


class TestCacheConfig:
    """Test the CacheConfig dataclass."""

    def test_cache_config_defaults(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.backend == "disk"
        assert config.default_ttl is None
        assert config.cache_dir == "./.cache"
        assert config.namespace is None
        assert config.debug is False
        assert config.llm_provider == "openai"
        assert config.image_key_replacement is True
        assert config.redis_url is None
        assert config.s3_bucket is None
        assert config._backend_instance is None

    def test_cache_config_environment_variables(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "CACHEFLOW_ENABLED": "false",
            "CACHEFLOW_BACKEND": "memory",
            "CACHEFLOW_DEFAULT_TTL": "3600",
            "CACHEFLOW_CACHE_DIR": "/tmp/custom_cache",
            "CACHEFLOW_NAMESPACE": "test_namespace",
            "CACHEFLOW_DEBUG": "true",
            "CACHEFLOW_LLM_PROVIDER": "anthropic",
            "CACHEFLOW_IMAGE_KEY_REPLACEMENT": "false",
            "CACHEFLOW_REDIS_URL": "redis://localhost:6379",
            "CACHEFLOW_S3_BUCKET": "my-cache-bucket",
        }

        with patch.dict(os.environ, env_vars):
            config = CacheConfig()

            assert config.enabled is False
            assert config.backend == "memory"
            assert config.default_ttl == 3600
            assert config.cache_dir == "/tmp/custom_cache"
            assert config.namespace == "test_namespace"
            assert config.debug is True
            assert config.llm_provider == "anthropic"
            assert config.image_key_replacement is False
            assert config.redis_url == "redis://localhost:6379"
            assert config.s3_bucket == "my-cache-bucket"

    def test_cache_config_bool_env_parsing(self):
        """Test boolean environment variable parsing."""
        true_values = ["true", "1", "yes", "on", "TRUE", "True"]
        false_values = ["false", "0", "no", "off", "FALSE", "False", ""]

        for value in true_values:
            with patch.dict(os.environ, {"CACHEFLOW_ENABLED": value}):
                config = CacheConfig()
                assert config.enabled is True

        for value in false_values:
            with patch.dict(os.environ, {"CACHEFLOW_ENABLED": value}):
                config = CacheConfig()
                assert config.enabled is False

    def test_cache_config_int_env_parsing(self):
        """Test integer environment variable parsing."""
        # Valid integer
        with patch.dict(os.environ, {"CACHEFLOW_DEFAULT_TTL": "3600"}):
            config = CacheConfig()
            assert config.default_ttl == 3600

        # Invalid integer - should use default
        with patch.dict(os.environ, {"CACHEFLOW_DEFAULT_TTL": "invalid"}):
            config = CacheConfig()
            assert config.default_ttl is None

    def test_cache_config_get_backend_memory(self):
        """Test getting memory backend."""
        config = CacheConfig(backend="memory")
        backend = config.get_backend()

        assert isinstance(backend, MemoryCacheBackend)
        assert config._backend_instance is backend

        # Second call should return same instance
        backend2 = config.get_backend()
        assert backend2 is backend

    def test_cache_config_get_backend_disk(self):
        """Test getting disk backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(backend="disk", cache_dir=temp_dir)
            backend = config.get_backend()

            assert isinstance(backend, DiskCacheBackend)
            assert backend.cache_dir == temp_dir
            assert config._backend_instance is backend

    def test_cache_config_reset_backend(self):
        """Test resetting backend instance."""
        config = CacheConfig(backend="memory")
        backend1 = config.get_backend()

        config.reset_backend()
        assert config._backend_instance is None

        backend2 = config.get_backend()
        assert backend2 is not backend1

    def test_cache_config_invalid_backend(self):
        """Test error with invalid backend type."""
        config = CacheConfig(backend="invalid_backend")

        with pytest.raises(ValueError, match="Unknown backend: invalid_backend"):
            config.get_backend()


class TestCreateBackend:
    """Test the create_backend function."""

    def test_create_backend_memory(self):
        """Test creating memory backend."""
        backend = create_backend("memory")
        assert isinstance(backend, MemoryCacheBackend)

    def test_create_backend_disk(self):
        """Test creating disk backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = create_backend("disk", cache_dir=temp_dir)
            assert isinstance(backend, DiskCacheBackend)
            assert backend.cache_dir == temp_dir

    def test_create_backend_disk_default_dir(self):
        """Test creating disk backend with default directory."""
        backend = create_backend("disk")
        assert isinstance(backend, DiskCacheBackend)
        assert backend.cache_dir == "./.cache"

    def test_create_backend_unknown(self):
        """Test error with unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            create_backend("unknown")


class TestGlobalConfiguration:
    """Test global configuration functions."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_get_config_returns_global_instance(self):
        """Test that get_config returns the global configuration instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2
        assert isinstance(config1, CacheConfig)

    def test_configure_updates_global_config(self):
        """Test that configure updates the global configuration."""
        initial_config = get_config()
        assert initial_config.enabled is True
        assert initial_config.backend == "disk"

        configure(enabled=False, backend="memory", debug=True)

        updated_config = get_config()
        assert updated_config is initial_config  # Same instance
        assert updated_config.enabled is False
        assert updated_config.backend == "memory"
        assert updated_config.debug is True

    def test_configure_invalid_key(self):
        """Test error when configuring with invalid key."""
        with pytest.raises(ValueError, match="Unknown configuration key: invalid_key"):
            configure(invalid_key="value")

    def test_configure_resets_backend_on_backend_change(self):
        """Test that configure resets backend when backend changes."""
        config = get_config()

        # Get initial backend
        backend1 = config.get_backend()
        assert isinstance(backend1, DiskCacheBackend)

        # Change backend
        configure(backend="memory")

        # Backend instance should be reset
        assert config._backend_instance is None

        # New backend should be different type
        backend2 = config.get_backend()
        assert isinstance(backend2, MemoryCacheBackend)
        assert backend2 is not backend1

    def test_configure_resets_backend_on_cache_dir_change(self):
        """Test that configure resets backend when cache_dir changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = get_config()

            # Get initial backend
            config.get_backend()

            # Change cache directory
            configure(cache_dir=temp_dir)

            # Backend instance should be reset
            assert config._backend_instance is None

            # New backend should use new directory
            backend2 = config.get_backend()
            assert backend2.cache_dir == temp_dir

    def test_configure_doesnt_reset_backend_on_other_changes(self):
        """Test that configure doesn't reset backend for non-backend changes."""
        config = get_config()

        # Get initial backend
        backend1 = config.get_backend()

        # Change non-backend settings
        configure(enabled=False, debug=True, namespace="test")

        # Backend instance should not be reset
        assert config._backend_instance is backend1

    def test_reset_config(self):
        """Test resetting configuration to defaults."""
        # Modify configuration
        configure(enabled=False, backend="memory", debug=True)
        config = get_config()
        assert config.enabled is False
        assert config.backend == "memory"
        assert config.debug is True

        # Reset configuration
        reset_config()

        # Should get new default configuration
        new_config = get_config()
        assert new_config is not config  # Different instance
        assert new_config.enabled is True
        assert new_config.backend == "disk"
        assert new_config.debug is False

    def test_multiple_configure_calls(self):
        """Test multiple configure calls."""
        configure(enabled=False)
        configure(backend="memory")
        configure(debug=True, namespace="test")

        config = get_config()
        assert config.enabled is False
        assert config.backend == "memory"
        assert config.debug is True
        assert config.namespace == "test"

    def test_configure_with_kwargs(self):
        """Test configure with various kwargs."""
        configure(
            enabled=True,
            backend="disk",
            default_ttl=3600,
            debug=False,
            namespace="production",
            llm_provider="anthropic",
        )

        config = get_config()
        assert config.enabled is True
        assert config.backend == "disk"
        assert config.default_ttl == 3600
        assert config.debug is False
        assert config.namespace == "production"
        assert config.llm_provider == "anthropic"


class TestEnvironmentIntegration:
    """Test integration between environment variables and global config."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_environment_override_at_startup(self):
        """Test that environment variables are loaded at startup."""
        env_vars = {
            "CACHEFLOW_ENABLED": "false",
            "CACHEFLOW_BACKEND": "memory",
            "CACHEFLOW_DEBUG": "true",
        }

        with patch.dict(os.environ, env_vars):
            # Reset to pick up environment variables
            reset_config()
            config = get_config()

            assert config.enabled is False
            assert config.backend == "memory"
            assert config.debug is True

    def test_configure_overrides_environment(self):
        """Test that configure() overrides environment variables."""
        env_vars = {
            "CACHEFLOW_ENABLED": "false",
            "CACHEFLOW_BACKEND": "memory",
        }

        with patch.dict(os.environ, env_vars):
            reset_config()
            config = get_config()

            # Initially from environment
            assert config.enabled is False
            assert config.backend == "memory"

            # Override with configure
            configure(enabled=True, backend="disk")

            assert config.enabled is True
            assert config.backend == "disk"

    def test_environment_variable_types(self):
        """Test different environment variable types."""
        env_vars = {
            "CACHEFLOW_ENABLED": "1",  # Boolean as "1"
            "CACHEFLOW_DEBUG": "yes",  # Boolean as "yes"
            "CACHEFLOW_DEFAULT_TTL": "7200",  # Integer
            "CACHEFLOW_CACHE_DIR": "/custom/path",  # String
        }

        with patch.dict(os.environ, env_vars):
            reset_config()
            config = get_config()

            assert config.enabled is True
            assert config.debug is True
            assert config.default_ttl == 7200
            assert config.cache_dir == "/custom/path"


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_empty_environment_variables(self):
        """Test behavior with empty environment variables."""
        env_vars = {
            "CACHEFLOW_ENABLED": "",
            "CACHEFLOW_DEFAULT_TTL": "",
            "CACHEFLOW_CACHE_DIR": "",
        }

        with patch.dict(os.environ, env_vars):
            reset_config()
            config = get_config()

            # Empty string should be treated as False for boolean
            assert config.enabled is False
            # Empty string should use default for int
            assert config.default_ttl is None
            # Empty string should be used as-is for string
            assert config.cache_dir == ""

    def test_concurrent_backend_access(self):
        """Test concurrent access to backend doesn't create multiple instances."""
        config = get_config()

        # Multiple calls should return same instance
        backends = [config.get_backend() for _ in range(10)]

        # All should be the same instance
        first_backend = backends[0]
        for backend in backends:
            assert backend is first_backend

    def test_config_immutability_after_backend_creation(self):
        """Test that backend changes after creation work correctly."""
        config = get_config()

        # Create backend
        backend1 = config.get_backend()
        assert isinstance(backend1, DiskCacheBackend)

        # Store reference to verify it's the same
        original_backend = config._backend_instance

        # Change backend type
        configure(backend="memory")

        # Backend should be reset
        assert config._backend_instance is None

        # New backend should be different
        backend2 = config.get_backend()
        assert isinstance(backend2, MemoryCacheBackend)
        assert backend2 is not original_backend

    def test_invalid_ttl_environment_variable(self):
        """Test handling of invalid TTL environment variable."""
        invalid_ttl_values = ["not_a_number", "3.14", "infinity"]

        for invalid_value in invalid_ttl_values:
            env_vars = {"CACHEFLOW_DEFAULT_TTL": invalid_value}

            with patch.dict(os.environ, env_vars):
                reset_config()
                config = get_config()

                # Should use default value for invalid TTL
                assert config.default_ttl is None

    def test_partial_environment_configuration(self):
        """Test configuration with only some environment variables set."""
        env_vars = {
            "CACHEFLOW_ENABLED": "true",
            "CACHEFLOW_DEBUG": "true",
            # Other variables not set
        }

        with patch.dict(os.environ, env_vars, clear=False):
            reset_config()
            config = get_config()

            # Set variables should be used
            assert config.enabled is True
            assert config.debug is True

            # Unset variables should use defaults
            assert config.backend == "disk"
            assert config.default_ttl is None
            assert config.namespace is None

    def test_config_repr_and_str(self):
        """Test string representation of config."""
        config = CacheConfig(enabled=False, backend="memory")

        # These should not raise exceptions
        str_repr = str(config)
        repr_str = repr(config)

        assert isinstance(str_repr, str)
        assert isinstance(repr_str, str)
        assert "CacheConfig" in repr_str
        assert "enabled" in repr_str or "enabled" in str_repr
