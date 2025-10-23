"""Unit tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from ali.config.settings import AppSettings, OllamaSettings, VisionSettings, get_settings


class TestOllamaSettings:
    """Test Ollama configuration."""

    def test_should_use_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        settings = OllamaSettings()

        # Assert
        assert settings.host == "http://localhost:11434"
        assert settings.model == "llama2"
        assert settings.timeout == 120
        assert settings.temperature == 0.7
        assert settings.context_window == 4096

    def test_should_accept_custom_values(self) -> None:
        """Test that custom values can be set."""
        # Arrange & Act
        settings = OllamaSettings(
            host="http://custom:8080",
            model="llama3",
            temperature=0.9,
        )

        # Assert
        assert settings.host == "http://custom:8080"
        assert settings.model == "llama3"
        assert settings.temperature == 0.9

    def test_should_validate_temperature_range(self) -> None:
        """Test that temperature is validated within range."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            OllamaSettings(temperature=3.0)

        with pytest.raises(Exception):
            OllamaSettings(temperature=-1.0)


class TestVisionSettings:
    """Test vision configuration."""

    def test_should_use_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        settings = VisionSettings()

        # Assert
        assert settings.screenshot_quality == 85
        assert settings.max_image_size == (1920, 1080)
        assert settings.vision_model == "llava"

    def test_should_validate_quality_range(self) -> None:
        """Test that quality is validated within range."""
        # Arrange & Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            VisionSettings(screenshot_quality=101)

        with pytest.raises(Exception):
            VisionSettings(screenshot_quality=0)


class TestAppSettings:
    """Test main application settings."""

    def test_should_use_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        settings = AppSettings()

        # Assert
        assert settings.app_name == "ALI"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_should_create_nested_settings(self) -> None:
        """Test that nested settings are initialized."""
        # Arrange & Act
        settings = AppSettings()

        # Assert
        assert isinstance(settings.ollama, OllamaSettings)
        assert isinstance(settings.vision, VisionSettings)

    def test_should_create_directories(self) -> None:
        """Test that ensure_directories creates required paths."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AppSettings(
                config_dir=Path(tmpdir) / "config",
                data_dir=Path(tmpdir) / "data",
                cache_dir=Path(tmpdir) / "cache",
            )

            # Act
            settings.ensure_directories()

            # Assert
            assert settings.config_dir.exists()
            assert settings.data_dir.exists()
            assert settings.cache_dir.exists()

    def test_should_have_valid_default_paths(self) -> None:
        """Test that default paths are set."""
        # Arrange & Act
        settings = AppSettings()

        # Assert
        assert settings.config_dir.is_absolute()
        assert settings.data_dir.is_absolute()
        assert settings.cache_dir.is_absolute()
        assert "ali" in str(settings.config_dir).lower()


class TestGetSettings:
    """Test settings singleton."""

    def test_should_return_same_instance(self) -> None:
        """Test that get_settings returns singleton instance."""
        # Arrange & Act
        settings1 = get_settings()
        settings2 = get_settings()

        # Assert
        assert settings1 is settings2
