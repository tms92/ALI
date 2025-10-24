"""Tests for interactive chat session."""

from unittest.mock import Mock

import pytest

from ali.core.chat import ChatSession


class TestChatSession:
    """Test suite for ChatSession class."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create mock Ollama client."""
        return Mock()

    @pytest.fixture
    def session(self, mock_client: Mock) -> ChatSession:
        """Create ChatSession with mock client."""
        return ChatSession(client=mock_client, model="llama2")

    def test_should_initialize_with_client(self, mock_client: Mock) -> None:
        """ChatSession should initialize with an OllamaClient."""
        # Arrange & Act
        session = ChatSession(client=mock_client, model="llama2")

        # Assert
        assert session.client == mock_client
        assert session.model == "llama2"
        assert session.history == []

    def test_should_initialize_with_system_prompt(self, mock_client: Mock) -> None:
        """ChatSession should accept optional system prompt."""
        # Arrange
        system_prompt = "You are a helpful assistant."

        # Act
        session = ChatSession(client=mock_client, model="llama2", system_prompt=system_prompt)

        # Assert
        assert session.system_prompt == system_prompt

    def test_should_initialize_with_empty_history(self, session: ChatSession) -> None:
        """ChatSession should start with empty message history."""
        # Arrange & Act (via fixture)
        # Assert
        assert isinstance(session.history, list)
        assert len(session.history) == 0

    def test_should_send_message_and_get_response(self, mock_client: Mock) -> None:
        """ChatSession should send message and return AI response."""
        # Arrange
        mock_client.chat.return_value = "Hello! How can I help you?"
        session = ChatSession(client=mock_client, model="llama2")

        # Act
        response = session.send_message("Hi there!")

        # Assert
        assert response == "Hello! How can I help you?"
        mock_client.chat.assert_called_once()

    def test_should_add_messages_to_history(self, mock_client: Mock) -> None:
        """ChatSession should track conversation history."""
        # Arrange
        mock_client.chat.return_value = "I'm doing well, thanks!"
        session = ChatSession(client=mock_client, model="llama2")

        # Act
        session.send_message("How are you?")

        # Assert
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "How are you?"}
        assert session.history[1] == {
            "role": "assistant",
            "content": "I'm doing well, thanks!",
        }

    def test_should_detect_exit_command(self, session: ChatSession) -> None:
        """ChatSession should recognize /exit and /quit commands."""
        # Arrange & Act & Assert
        assert session.is_command("/exit") is True
        assert session.is_command("/quit") is True
        assert session.is_command("regular message") is False

    def test_should_parse_exit_command(self, session: ChatSession) -> None:
        """ChatSession should parse exit commands correctly."""
        # Arrange & Act
        result_exit = session.parse_command("/exit")
        result_quit = session.parse_command("/quit")

        # Assert
        assert result_exit == {"command": "exit", "args": []}
        assert result_quit == {"command": "quit", "args": []}

    def test_should_parse_help_command(self, session: ChatSession) -> None:
        """ChatSession should parse /help command."""
        # Arrange & Act
        result = session.parse_command("/help")

        # Assert
        assert result == {"command": "help", "args": []}

    def test_should_parse_clear_command(self, session: ChatSession) -> None:
        """ChatSession should parse /clear command."""
        # Arrange & Act
        result = session.parse_command("/clear")

        # Assert
        assert result == {"command": "clear", "args": []}

    def test_should_parse_model_command_with_argument(self, session: ChatSession) -> None:
        """ChatSession should parse /model command with model name."""
        # Arrange & Act
        result = session.parse_command("/model llama3")

        # Assert
        assert result == {"command": "model", "args": ["llama3"]}

    def test_should_parse_history_command(self, session: ChatSession) -> None:
        """ChatSession should parse /history command."""
        # Arrange & Act
        result = session.parse_command("/history")

        # Assert
        assert result == {"command": "history", "args": []}

    def test_should_clear_history(self, mock_client: Mock) -> None:
        """ChatSession should clear conversation history on request."""
        # Arrange
        mock_client.chat.return_value = "Response"
        session = ChatSession(client=mock_client, model="llama2")
        session.send_message("First message")
        session.send_message("Second message")

        # Act
        assert len(session.history) == 4  # 2 user + 2 assistant
        session.clear_history()

        # Assert
        assert len(session.history) == 0

    def test_should_change_model(self, session: ChatSession) -> None:
        """ChatSession should allow changing the active model."""
        # Arrange
        assert session.model == "llama2"

        # Act
        session.change_model("llama3")

        # Assert
        assert session.model == "llama3"

    def test_should_get_help_text(self, session: ChatSession) -> None:
        """ChatSession should provide help text for available commands."""
        # Arrange & Act
        help_text = session.get_help()

        # Assert
        assert "/exit" in help_text or "/quit" in help_text
        assert "/help" in help_text
        assert "/clear" in help_text
        assert "/model" in help_text
        assert "/history" in help_text

    def test_should_format_history_for_display(self, mock_client: Mock) -> None:
        """ChatSession should format conversation history for display."""
        # Arrange
        mock_client.chat.return_value = "Hello!"
        session = ChatSession(client=mock_client, model="llama2")
        session.send_message("Hi")

        # Act
        history_display = session.format_history()

        # Assert
        assert "User:" in history_display or "You:" in history_display
        assert "Assistant:" in history_display or "ALI:" in history_display
        assert "Hi" in history_display
        assert "Hello!" in history_display

    def test_should_handle_empty_message(self, session: ChatSession) -> None:
        """ChatSession should handle empty messages gracefully."""
        # Arrange & Act
        response = session.send_message("")

        # Assert
        assert response is None or "empty" in response.lower()
        session.client.chat.assert_not_called()

    def test_should_handle_api_error(self, mock_client: Mock) -> None:
        """ChatSession should handle client errors gracefully."""
        # Arrange
        mock_client.chat.side_effect = Exception("API Error")
        session = ChatSession(client=mock_client, model="llama2")

        # Act & Assert
        with pytest.raises(Exception):
            session.send_message("Test message")

    def test_should_pass_system_prompt_to_client(self, mock_client: Mock) -> None:
        """ChatSession should pass system prompt when sending messages."""
        # Arrange
        mock_client.chat.return_value = "Response"
        system_prompt = "You are ALI"
        session = ChatSession(client=mock_client, model="llama2", system_prompt=system_prompt)

        # Act
        session.send_message("Hello")

        # Assert
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs.get("system_prompt") == system_prompt

    def test_should_return_none_for_invalid_command(self, session: ChatSession) -> None:
        """ChatSession should handle invalid commands gracefully."""
        # Arrange & Act
        result = session.parse_command("/invalid_command")

        # Assert
        assert result == {"command": "unknown", "args": ["/invalid_command"]}
