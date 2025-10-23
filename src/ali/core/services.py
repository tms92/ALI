"""Service management for external dependencies."""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


class OllamaService:
    """Manage Ollama service lifecycle."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        """Initialize Ollama service manager.

        Args:
            host: Ollama API host URL
        """
        self.host = host
        self.base_url = host.rstrip("/")

    def is_installed(self) -> bool:
        """Check if Ollama is installed on the system.

        Returns:
            True if Ollama executable is found, False otherwise
        """
        ollama_path = shutil.which("ollama")
        if ollama_path:
            logger.debug(f"Ollama found at: {ollama_path}")
            return True
        logger.warning("Ollama executable not found in PATH")
        return False

    def is_running(self) -> bool:
        """Check if Ollama service is running and responding.

        Returns:
            True if service is running and accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.debug("Ollama service is running")
                return True
        except requests.exceptions.RequestException as e:
            logger.debug(f"Ollama not responding: {e}")
        return False

    def start(self, wait_timeout: int = 10) -> bool:
        """Start Ollama service if not running.

        Args:
            wait_timeout: Maximum seconds to wait for service to start

        Returns:
            True if service started successfully, False otherwise
        """
        if self.is_running():
            logger.info("Ollama already running")
            return True

        if not self.is_installed():
            logger.error("Cannot start Ollama: not installed")
            return False

        try:
            logger.info("Starting Ollama service...")

            # Start Ollama serve in background
            # On Windows, use CREATE_NO_WINDOW flag to avoid showing console
            if subprocess.os.name == "nt":  # Windows
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:  # Unix-like
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            # Wait for service to be ready
            start_time = time.time()
            while time.time() - start_time < wait_timeout:
                if self.is_running():
                    logger.info("Ollama service started successfully")
                    return True
                time.sleep(0.5)

            logger.error(f"Ollama failed to start within {wait_timeout}s")
            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

    def ensure_running(self, auto_start: bool = True) -> bool:
        """Ensure Ollama service is running, optionally starting it.

        Args:
            auto_start: If True, attempt to start service if not running

        Returns:
            True if service is running, False otherwise
        """
        if self.is_running():
            return True

        if not self.is_installed():
            logger.error(
                "Ollama is not installed. Please install from: https://ollama.com/download"
            )
            return False

        if auto_start:
            logger.info("Ollama not running, attempting to start...")
            return self.start()

        logger.error("Ollama is not running. Please start it with: ollama serve")
        return False

    def list_models(self) -> list[str]:
        """List available models from running Ollama instance.

        Returns:
            List of model names, empty if service not available
        """
        if not self.is_running():
            logger.warning("Cannot list models: Ollama not running")
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            logger.debug(f"Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def has_model(self, model_name: str) -> bool:
        """Check if a specific model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available, False otherwise
        """
        models = self.list_models()
        return model_name in models

    def suggest_model(self) -> Optional[str]:
        """Suggest a model to use based on what's available.

        Returns:
            Suggested model name, or None if no models available
        """
        models = self.list_models()
        if not models:
            return None

        # Preference order for default models
        preferred = ["llama3.2", "llama3.1", "llama3", "llama2", "mistral"]

        for pref in preferred:
            for model in models:
                if model.startswith(pref):
                    return model

        # Return first available model if no preferred found
        return models[0]
