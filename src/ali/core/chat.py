"""Interactive chat session for conversational AI."""

from typing import Any

from loguru import logger

from ali.llm.client import OllamaClient


class ChatSession:
    """Manages an interactive chat session with conversation history."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize chat session.

        Args:
            client: OllamaClient instance for LLM communication
            model: Name of the model to use
            system_prompt: Optional system prompt for the assistant
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or "You are ALI, a helpful local AI assistant."
        self.history: list[dict[str, str]] = []

    def send_message(self, message: str) -> str | None:
        """Send a message and get response from the assistant.

        Args:
            message: User message to send

        Returns:
            Assistant's response, or None if message is empty
        """
        if not message or not message.strip():
            return None

        # Add user message to history
        self.history.append({"role": "user", "content": message})

        try:
            # Get response from LLM
            response = self.client.chat(
                message=message,
                model=self.model,
                system_prompt=self.system_prompt,
            )

            # Add assistant response to history
            self.history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    def is_command(self, message: str) -> bool:
        """Check if message is a command.

        Args:
            message: Message to check

        Returns:
            True if message starts with '/', False otherwise
        """
        return message.strip().startswith("/")

    def parse_command(self, message: str) -> dict[str, Any]:
        """Parse a command message.

        Args:
            message: Command message (starting with '/')

        Returns:
            Dictionary with 'command' and 'args' keys
        """
        parts = message.strip().split()
        if not parts:
            return {"command": "unknown", "args": []}

        command_name = parts[0][1:]  # Remove leading '/'
        args = parts[1:] if len(parts) > 1 else []

        # Map command aliases
        if command_name in ["exit", "quit"]:
            return {"command": command_name, "args": args}
        elif command_name == "help":
            return {"command": "help", "args": args}
        elif command_name == "clear":
            return {"command": "clear", "args": args}
        elif command_name == "model":
            return {"command": "model", "args": args}
        elif command_name == "history":
            return {"command": "history", "args": args}
        else:
            return {"command": "unknown", "args": [message]}

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")

    def change_model(self, model: str) -> None:
        """Change the active model.

        Args:
            model: Name of the new model to use
        """
        old_model = self.model
        self.model = model
        logger.info(f"Changed model from {old_model} to {model}")

    def get_help(self) -> str:
        """Get help text for available commands.

        Returns:
            Help text describing all available commands
        """
        return """
Available commands:

  /help              Show this help message
  /exit, /quit       Exit the chat session
  /clear             Clear conversation history
  /model <name>      Switch to a different model
  /history           Show conversation history

Type your message and press Enter to chat with ALI.
        """.strip()

    def format_history(self) -> str:
        """Format conversation history for display.

        Returns:
            Formatted string of conversation history
        """
        if not self.history:
            return "No conversation history yet."

        lines = []
        for entry in self.history:
            role = entry["role"]
            content = entry["content"]

            if role == "user":
                lines.append(f"You: {content}")
            elif role == "assistant":
                lines.append(f"ALI: {content}")

        return "\n".join(lines)
