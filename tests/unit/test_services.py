"""Unit tests for service management."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from ali.core.services import OllamaService


class TestOllamaService:
    """Test Ollama service manager."""

    @pytest.fixture
    def service(self) -> OllamaService:
        """Create OllamaService instance."""
        return OllamaService("http://test:11434")

    def test_should_initialize_with_host(self) -> None:
        """Test that service initializes with correct host."""
        # Arrange & Act
        service = OllamaService("http://custom:8080")

        # Assert
        assert service.host == "http://custom:8080"
        assert service.base_url == "http://custom:8080"

    def test_should_strip_trailing_slash_from_url(self) -> None:
        """Test that trailing slash is removed from base URL."""
        # Arrange & Act
        service = OllamaService("http://test:11434/")

        # Assert
        assert service.base_url == "http://test:11434"

    @patch("ali.core.services.shutil.which")
    def test_should_detect_installed_ollama(self, mock_which: Mock, service: OllamaService) -> None:
        """Test that is_installed returns True when ollama found."""
        # Arrange
        mock_which.return_value = "/usr/bin/ollama"

        # Act
        result = service.is_installed()

        # Assert
        assert result is True
        mock_which.assert_called_once_with("ollama")

    @patch("ali.core.services.shutil.which")
    def test_should_detect_missing_ollama(self, mock_which: Mock, service: OllamaService) -> None:
        """Test that is_installed returns False when ollama not found."""
        # Arrange
        mock_which.return_value = None

        # Act
        result = service.is_installed()

        # Assert
        assert result is False

    @patch("ali.core.services.requests.get")
    def test_should_detect_running_service(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that is_running returns True when service responds."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Act
        result = service.is_running()

        # Assert
        assert result is True
        mock_get.assert_called_once_with("http://test:11434/api/tags", timeout=2)

    @patch("ali.core.services.requests.get")
    def test_should_detect_stopped_service(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that is_running returns False when service not responding."""
        # Arrange
        mock_get.side_effect = requests.exceptions.ConnectionError()

        # Act
        result = service.is_running()

        # Assert
        assert result is False

    @patch("ali.core.services.requests.get")
    def test_should_return_empty_list_when_service_not_running(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that list_models returns empty list when service down."""
        # Arrange
        mock_get.side_effect = requests.exceptions.ConnectionError()

        # Act
        models = service.list_models()

        # Assert
        assert models == []

    @patch("ali.core.services.requests.get")
    def test_should_list_available_models(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that list_models returns model names."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "llama3"},
                {"name": "mistral"},
            ]
        }
        mock_get.return_value = mock_response

        # Act
        models = service.list_models()

        # Assert
        assert models == ["llama2", "llama3", "mistral"]

    @patch("ali.core.services.requests.get")
    def test_should_check_if_model_exists(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that has_model correctly identifies available models."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2"}, {"name": "mistral"}]
        }
        mock_get.return_value = mock_response

        # Act & Assert
        assert service.has_model("llama2") is True
        assert service.has_model("llama3") is False

    @patch("ali.core.services.requests.get")
    def test_should_suggest_preferred_model(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that suggest_model prefers llama3 over llama2."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "llama3.1"},
                {"name": "mistral"},
            ]
        }
        mock_get.return_value = mock_response

        # Act
        suggested = service.suggest_model()

        # Assert
        assert suggested == "llama3.1"

    @patch("ali.core.services.requests.get")
    def test_should_return_none_when_no_models_available(
        self, mock_get: Mock, service: OllamaService
    ) -> None:
        """Test that suggest_model returns None when no models."""
        # Arrange
        mock_get.side_effect = requests.exceptions.ConnectionError()

        # Act
        suggested = service.suggest_model()

        # Assert
        assert suggested is None

    @patch("ali.core.services.OllamaService.is_running")
    def test_should_return_true_when_already_running(
        self, mock_is_running: Mock, service: OllamaService
    ) -> None:
        """Test that ensure_running returns True if already running."""
        # Arrange
        mock_is_running.return_value = True

        # Act
        result = service.ensure_running()

        # Assert
        assert result is True

    @patch("ali.core.services.OllamaService.is_installed")
    @patch("ali.core.services.OllamaService.is_running")
    def test_should_return_false_when_not_installed(
        self, mock_is_running: Mock, mock_is_installed: Mock, service: OllamaService
    ) -> None:
        """Test that ensure_running returns False if not installed."""
        # Arrange
        mock_is_running.return_value = False
        mock_is_installed.return_value = False

        # Act
        result = service.ensure_running()

        # Assert
        assert result is False
