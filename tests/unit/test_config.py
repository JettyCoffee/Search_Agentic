"""
Unit tests for the configuration module.
"""

import pytest
import os
from unittest.mock import patch, mock_open
from pathlib import Path

from search_agent.utils.config import Config
from search_agent.exceptions.custom_exceptions import ConfigurationError


class TestConfig:
    """Test cases for the Config class."""
    
    def test_initialization_default(self):
        """Test Config initialization with default values."""
        config = Config()
        
        assert config.log_level == "INFO"
        assert config.cache_type == "memory"
        assert config.rate_limit_enabled is True
        assert config.max_results_per_source == 10
    
    def test_initialization_with_custom_config(self):
        """Test Config initialization with custom configuration."""
        custom_config = {
            "log_level": "DEBUG",
            "cache_type": "file",
            "max_results_per_source": 20
        }
        
        config = Config(custom_config)
        
        assert config.log_level == "DEBUG"
        assert config.cache_type == "file"
        assert config.max_results_per_source == 20
    
    @patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test_google_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "LOG_LEVEL": "WARNING"
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = Config()
        
        assert config.google_api_key == "test_google_key"
        assert config.gemini_api_key == "test_gemini_key"
        assert config.log_level == "WARNING"
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        mock_config_content = """
        GOOGLE_API_KEY=file_google_key
        BRAVE_API_KEY=file_brave_key
        LOG_LEVEL=ERROR
        """
        
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            with patch("os.path.exists", return_value=True):
                config = Config()
                config.load_from_file(".env")
                
                assert config.google_api_key == "file_google_key"
                assert config.brave_api_key == "file_brave_key"
                assert config.log_level == "ERROR"
    
    def test_load_from_nonexistent_file(self):
        """Test loading from a file that doesn't exist."""
        config = Config()
        
        with patch("os.path.exists", return_value=False):
            # Should not raise an exception
            config.load_from_file("nonexistent.env")
    
    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = Config({
            "google_api_key": "valid_key",
            "gemini_api_key": "valid_key",
            "log_level": "INFO"
        })
        
        assert config.validate() is True
    
    def test_validate_missing_required_keys(self):
        """Test validation with missing required API keys."""
        config = Config()
        
        with pytest.raises(ConfigurationError, match="Missing required API keys"):
            config.validate()
    
    def test_validate_invalid_log_level(self):
        """Test validation with invalid log level."""
        config = Config({
            "google_api_key": "valid_key",
            "gemini_api_key": "valid_key", 
            "log_level": "INVALID"
        })
        
        with pytest.raises(ConfigurationError, match="Invalid log level"):
            config.validate()
    
    def test_validate_invalid_cache_type(self):
        """Test validation with invalid cache type."""
        config = Config({
            "google_api_key": "valid_key",
            "gemini_api_key": "valid_key",
            "cache_type": "invalid_cache"
        })
        
        with pytest.raises(ConfigurationError, match="Invalid cache type"):
            config.validate()
    
    def test_validate_invalid_max_results(self):
        """Test validation with invalid max results."""
        config = Config({
            "google_api_key": "valid_key",
            "gemini_api_key": "valid_key",
            "max_results_per_source": -1
        })
        
        with pytest.raises(ConfigurationError, match="max_results_per_source must be positive"):
            config.validate()
    
    def test_get_api_key_existing(self):
        """Test getting an existing API key."""
        config = Config({"google_api_key": "test_key"})
        
        assert config.get_api_key("google") == "test_key"
    
    def test_get_api_key_nonexistent(self):
        """Test getting a non-existent API key."""
        config = Config()
        
        assert config.get_api_key("nonexistent") is None
    
    def test_set_api_key(self):
        """Test setting an API key."""
        config = Config()
        config.set_api_key("test_service", "test_key")
        
        assert config.get_api_key("test_service") == "test_key"
    
    def test_get_config_dict(self):
        """Test getting configuration as dictionary."""
        custom_config = {
            "log_level": "DEBUG",
            "cache_type": "file",
            "google_api_key": "test_key"
        }
        
        config = Config(custom_config)
        config_dict = config.get_config_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["log_level"] == "DEBUG"
        assert config_dict["cache_type"] == "file"
        # API keys should be masked
        assert config_dict["google_api_key"] == "****_key"
    
    def test_update_config(self):
        """Test updating configuration."""
        config = Config({"log_level": "INFO"})
        
        config.update_config({
            "log_level": "DEBUG",
            "cache_type": "file"
        })
        
        assert config.log_level == "DEBUG"
        assert config.cache_type == "file"
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config({
            "google_api_key": "test_key",
            "log_level": "DEBUG"
        })
        
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            config.save_to_file("test_config.env")
            
            # Verify file was written
            mock_file.assert_called_once_with("test_config.env", "w")
            
            # Check that content was written (API keys should be masked)
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "LOG_LEVEL=DEBUG" in written_content
            assert "GOOGLE_API_KEY=" in written_content
    
    def test_environment_override(self):
        """Test that environment variables override default values."""
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            config = Config({"log_level": "INFO"})
            
            # Environment should override the provided config
            assert config.log_level == "ERROR"
    
    def test_case_insensitive_keys(self):
        """Test that configuration keys are case insensitive."""
        config = Config({
            "Google_API_Key": "test_key",
            "LOG_level": "DEBUG"
        })
        
        assert config.google_api_key == "test_key"
        assert config.log_level == "DEBUG"
    
    def test_boolean_conversion(self):
        """Test conversion of string boolean values."""
        config = Config({
            "rate_limit_enabled": "false",
            "cache_enabled": "true"
        })
        
        assert config.rate_limit_enabled is False
        assert config.cache_enabled is True
    
    def test_integer_conversion(self):
        """Test conversion of string integer values."""
        config = Config({
            "max_results_per_source": "25",
            "cache_ttl": "3600"
        })
        
        assert config.max_results_per_source == 25
        assert config.cache_ttl == 3600
    
    def test_list_conversion(self):
        """Test conversion of comma-separated string to list."""
        config = Config({
            "default_sources": "wikipedia,google_search,arxiv"
        })
        
        expected_sources = ["wikipedia", "google_search", "arxiv"]
        assert config.default_sources == expected_sources
    
    def test_config_inheritance(self):
        """Test configuration inheritance and merging."""
        base_config = {
            "log_level": "INFO",
            "cache_type": "memory",
            "google_api_key": "base_key"
        }
        
        override_config = {
            "log_level": "DEBUG",
            "brave_api_key": "new_key"
        }
        
        config = Config(base_config)
        config.update_config(override_config)
        
        # Should merge both configs
        assert config.log_level == "DEBUG"  # Overridden
        assert config.cache_type == "memory"  # Inherited
        assert config.google_api_key == "base_key"  # Inherited
        assert config.brave_api_key == "new_key"  # New


class TestConfigEdgeCases:
    """Test edge cases and error conditions for Config."""
    
    def test_empty_config(self):
        """Test Config with empty configuration."""
        config = Config({})
        
        # Should use defaults
        assert config.log_level == "INFO"
        assert config.cache_type == "memory"
    
    def test_none_config(self):
        """Test Config with None configuration."""
        config = Config(None)
        
        # Should use defaults
        assert config.log_level == "INFO"
        assert config.cache_type == "memory"
    
    def test_invalid_config_type(self):
        """Test Config with invalid configuration type."""
        with pytest.raises(TypeError):
            Config("invalid_config_string")
    
    def test_api_key_validation_empty_string(self):
        """Test API key validation with empty strings."""
        config = Config({
            "google_api_key": "",
            "gemini_api_key": ""
        })
        
        with pytest.raises(ConfigurationError, match="Missing required API keys"):
            config.validate()
    
    def test_api_key_validation_whitespace(self):
        """Test API key validation with whitespace-only strings."""
        config = Config({
            "google_api_key": "   ",
            "gemini_api_key": "   "
        })
        
        with pytest.raises(ConfigurationError, match="Missing required API keys"):
            config.validate()
    
    def test_file_loading_permission_error(self):
        """Test file loading with permission error."""
        config = Config()
        
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with patch("os.path.exists", return_value=True):
                # Should not raise an exception, just log warning
                config.load_from_file(".env")
    
    def test_file_saving_permission_error(self):
        """Test file saving with permission error."""
        config = Config()
        
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                config.save_to_file("readonly.env")
    
    def test_special_characters_in_values(self):
        """Test configuration with special characters in values."""
        config = Config({
            "google_api_key": "key_with_special_chars!@#$%^&*()",
            "description": "A config with special chars: äöü"
        })
        
        assert config.google_api_key == "key_with_special_chars!@#$%^&*()"
        assert config.description == "A config with special chars: äöü"
