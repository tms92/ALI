"""Ollama LLM client wrapper."""

from collections.abc import AsyncIterator

import ollama
from loguru import logger

from ali.config.settings import OllamaSettings


class OllamaClient:
    """Wrapper for Ollama API client."""

    def __init__(self, settings: OllamaSettings | None = None) -> None:
        """Initialize Ollama client.

        Args:
            settings: Ollama configuration settings
        """
        self.settings = settings or OllamaSettings()
        self.client = ollama.Client(host=self.settings.host, timeout=self.settings.timeout)
        logger.info(f"Initialized Ollama client: {self.settings.host}")

    def chat(
        self,
        message: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a chat message and get response.

        Args:
            message: User message
            model: Model to use (defaults to settings)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (defaults to settings)

        Returns:
            Model response text
        """
        model = model or self.settings.model
        temperature = temperature if temperature is not None else self.settings.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        try:
            logger.debug(f"Sending chat request to {model}")
            response = self.client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            content: str = response["message"]["content"]
            logger.debug(f"Received response ({len(content)} chars)")
            return content

        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

    async def chat_stream(
        self,
        message: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Send a chat message and stream response.

        Args:
            message: User message
            model: Model to use (defaults to settings)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (defaults to settings)

        Yields:
            Response chunks
        """
        model = model or self.settings.model
        temperature = temperature if temperature is not None else self.settings.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        try:
            logger.debug(f"Sending streaming chat request to {model}")
            stream = self.client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                yield content

        except Exception as e:
            logger.error(f"Streaming chat request failed: {e}")
            raise

    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names
        """
        try:
            response = self.client.list()
            models = [model["model"] for model in response["models"]]
            logger.info(f"Found {len(models)} available models")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if pull succeeded, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model}")
            self.client.pull(model)
            logger.info(f"Successfully pulled model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
