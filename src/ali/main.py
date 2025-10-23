"""Main entry point for ALI application."""

import sys

from loguru import logger

from ali.config.settings import get_settings
from ali.core.services import OllamaService
from ali.llm.client import OllamaClient


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
    )

    # Add file handler
    log_file = settings.data_dir / "logs" / "ali.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )


def main() -> None:
    """Main application entry point."""
    setup_logging()
    settings = get_settings()

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    try:
        # Initialize Ollama service manager
        ollama_service = OllamaService(settings.ollama.host)

        # Ensure Ollama is running (auto-start if needed)
        logger.info("Checking Ollama service...")
        if not ollama_service.ensure_running(auto_start=True):
            logger.error("Failed to start Ollama service")
            logger.info("Please install Ollama from: https://ollama.com/download")
            sys.exit(1)

        # Check for available models
        models = ollama_service.list_models()
        if not models:
            logger.warning("No models found locally")
            logger.info("Pulling default model (llama2)...")
            logger.info("This may take a few minutes on first run...")
            # Note: We'll add auto-pull in next iteration
            logger.info("Please run: ollama pull llama2")
            sys.exit(1)

        logger.info(f"Available models: {', '.join(models)}")

        # Use suggested model or configured default
        suggested_model = ollama_service.suggest_model()
        model_to_use = suggested_model or settings.ollama.model

        if not ollama_service.has_model(model_to_use):
            logger.warning(f"Configured model '{model_to_use}' not found")
            logger.info(f"Using available model: {suggested_model}")
            model_to_use = suggested_model

        # Initialize Ollama client
        client = OllamaClient(settings.ollama)

        # Simple test interaction
        logger.info(f"Testing with model: {model_to_use}")
        response = client.chat(
            message="Hello! Please introduce yourself briefly.",
            system_prompt="You are ALI, a helpful local AI assistant.",
            model=model_to_use,
        )

        print("\n" + "=" * 50)
        print("ALI Response:")
        print("=" * 50)
        print(response)
        print("=" * 50 + "\n")

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
