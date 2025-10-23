"""Unit tests for Ollama LLM client."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ali.config.settings import OllamaSettings
from ali.llm.client import OllamaClient


class TestOllamaClient:
    """Test Ollama client."""

    @pytest.fixture
    def mock_ollama_client(self) -> Mock:
        """Create mock Ollama client."""
        return MagicMock()

    @pytest.fixture
    def settings(self) -> OllamaSettings:
        """Create test settings."""
        return OllamaSettings(
            host="http://test:11434",
            model="test-model",
            temperature=0.5,
        )

    @patch("ali.llm.client.ollama.Client")
    def test_should_initialize_with_settings(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that client initializes with correct settings."""
        # Arrange & Act
        client = OllamaClient(settings)

        # Assert
        mock_client_class.assert_called_once_with(
            host=settings.host,
            timeout=settings.timeout,
        )
        assert client.settings == settings

    @patch("ali.llm.client.ollama.Client")
    def test_should_initialize_with_default_settings(self, mock_client_class: Mock) -> None:
        """Test that client can initialize with default settings."""
        # Arrange & Act
        client = OllamaClient()

        # Assert
        assert client.settings is not None
        assert isinstance(client.settings, OllamaSettings)

    @patch("ali.llm.client.ollama.Client")
    def test_should_send_chat_message(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that chat sends message correctly."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Test response"}
        }

        client = OllamaClient(settings)

        # Act
        response = client.chat("Test message")

        # Assert
        assert response == "Test response"
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args.kwargs["model"] == settings.model
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["content"] == "Test message"

    @patch("ali.llm.client.ollama.Client")
    def test_should_include_system_prompt_when_provided(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that system prompt is included when provided."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Test response"}
        }

        client = OllamaClient(settings)

        # Act
        client.chat("Test message", system_prompt="You are a test assistant")

        # Assert
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a test assistant"
        assert messages[1]["role"] == "user"

    @patch("ali.llm.client.ollama.Client")
    def test_should_use_custom_model_when_provided(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that custom model overrides default."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Test response"}
        }

        client = OllamaClient(settings)

        # Act
        client.chat("Test message", model="custom-model")

        # Assert
        call_args = mock_client.chat.call_args
        assert call_args.kwargs["model"] == "custom-model"

    @patch("ali.llm.client.ollama.Client")
    def test_should_use_custom_temperature_when_provided(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that custom temperature overrides default."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "Test response"}
        }

        client = OllamaClient(settings)

        # Act
        client.chat("Test message", temperature=0.9)

        # Assert
        call_args = mock_client.chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.9

    @patch("ali.llm.client.ollama.Client")
    def test_should_raise_exception_on_error(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that exceptions are propagated."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.side_effect = Exception("Connection error")

        client = OllamaClient(settings)

        # Act & Assert
        with pytest.raises(Exception, match="Connection error"):
            client.chat("Test message")

    @patch("ali.llm.client.ollama.Client")
    def test_should_list_available_models(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that list_models returns available models."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.list.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "llama3"},
                {"name": "codellama"},
            ]
        }

        client = OllamaClient(settings)

        # Act
        models = client.list_models()

        # Assert
        assert models == ["llama2", "llama3", "codellama"]
        mock_client.list.assert_called_once()

    @patch("ali.llm.client.ollama.Client")
    def test_should_pull_model(
        self, mock_client_class: Mock, settings: OllamaSettings
    ) -> None:
        """Test that pull_model calls ollama pull."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = OllamaClient(settings)

        # Act
        client.pull_model("new-model")

        # Assert
        mock_client.pull.assert_called_once_with("new-model")
