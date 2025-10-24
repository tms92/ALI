"""Main entry point for ALI application."""

import sys

from loguru import logger

from ali.config.settings import get_settings
from ali.core.onboarding import OnboardingFlow
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


def needs_onboarding(settings: "ali.config.settings.AppSettings") -> bool:  # type: ignore[name-defined]  # noqa: F821, E501
    """Check if user needs to go through onboarding.

    Args:
        settings: Application settings

    Returns:
        True if onboarding is needed, False otherwise
    """
    # Check if onboarding has been completed (marker file)
    onboarding_complete_file = settings.config_dir / ".onboarding_complete"
    return not onboarding_complete_file.exists()


def main() -> None:
    """Main application entry point."""
    setup_logging()
    settings = get_settings()

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    try:
        # Check if onboarding is needed
        if needs_onboarding(settings):
            logger.info("First-time setup detected, starting onboarding...")
            onboarding = OnboardingFlow()
            result = onboarding.run()

            if not result.success:
                logger.error(f"Onboarding failed: {result.error_message}")
                print("\n✗ Setup incomplete. Please try again or install Ollama manually.")
                print("  Visit: https://ollama.com/download\n")
                sys.exit(1)

            # Mark onboarding as complete
            onboarding_complete_file = settings.config_dir / ".onboarding_complete"
            onboarding_complete_file.parent.mkdir(parents=True, exist_ok=True)
            onboarding_complete_file.write_text("")
            logger.info("Onboarding completed successfully")
        else:
            logger.info("Skipping onboarding (already configured)")

        # Initialize services
        ollama_service = OllamaService(settings.ollama.host)

        # Ensure Ollama is running
        if not ollama_service.ensure_running(auto_start=True):
            logger.error("Ollama service is not running")
            print("\n✗ Ollama is not running. Please start it or run setup again.\n")
            sys.exit(1)

        # Determine which model to use
        preference_file = settings.config_dir / "model_preference.txt"
        if preference_file.exists():
            model_to_use = preference_file.read_text().strip()
            logger.info(f"Using saved preference: {model_to_use}")
        else:
            # No preference saved - ask user to select
            logger.info("No model preference found, asking user to select")
            from ali.core.hardware import detect_hardware
            from ali.core.recommendations import get_installed_models, get_model_recommendations

            client = OllamaClient(settings.ollama)
            hardware = detect_hardware()
            installed_models = get_installed_models(client)
            recommendations = get_model_recommendations(hardware, installed_models)

            # Reuse onboarding flow for selection
            onboarding = OnboardingFlow()
            onboarding.display_recommendations(recommendations, hardware)
            selected = onboarding.select_model_interactive(recommendations)

            if not selected:
                logger.error("User cancelled model selection")
                print("\n✗ No model selected. Exiting.\n")
                sys.exit(1)

            model_to_use = selected

            # Ask if user wants to save this choice
            if onboarding.ask_save_preference(model_to_use):
                onboarding.save_preference(model_to_use)

        # Verify model exists
        if not ollama_service.has_model(model_to_use):
            logger.warning(f"Preferred model '{model_to_use}' not found")
            suggested = ollama_service.suggest_model()
            if not suggested:
                logger.error("No models available")
                print("\n✗ No models found. Please run setup or install a model.\n")
                sys.exit(1)
            model_to_use = suggested
            logger.info(f"Falling back to: {model_to_use}")

        # Initialize client
        client = OllamaClient(settings.ollama)

        # Test interaction
        logger.info(f"Testing ALI with model: {model_to_use}")
        print("\nAsking ALI to introduce itself...\n")

        response = client.chat(
            message="Hello! Please introduce yourself briefly in 2-3 sentences.",
            system_prompt="You are ALI, a helpful local AI assistant.",
            model=model_to_use,
        )

        print("=" * 70)
        print("  ALI Response")
        print("=" * 70)
        print(f"\n{response}\n")
        print("=" * 70 + "\n")

        logger.info("ALI is ready to use!")
        print("✓ ALI initialized successfully!\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Application error: {e}")
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
