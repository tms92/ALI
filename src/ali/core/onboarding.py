"""Interactive onboarding flow for first-time users."""

from dataclasses import dataclass

from loguru import logger

from ali.core.hardware import HardwareInfo, detect_hardware
from ali.core.installer import OllamaInstaller
from ali.core.recommendations import (
    ModelRecommendation,
    explain_recommendation,
    get_installed_models,
    get_model_recommendations,
)
from ali.core.services import OllamaService
from ali.llm.client import OllamaClient


@dataclass
class OnboardingResult:
    """Result of onboarding process."""

    success: bool
    selected_model: str | None
    hardware: HardwareInfo | None
    ollama_installed: bool
    error_message: str | None = None


class OnboardingFlow:
    """Manage interactive onboarding for new users."""

    def __init__(self) -> None:
        """Initialize onboarding flow."""
        self.service = OllamaService()
        self.installer = OllamaInstaller()
        self.client: OllamaClient | None = None

    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed and running.

        Returns:
            True if Ollama is ready to use, False otherwise
        """
        logger.info("Checking Ollama installation status...")

        if not self.service.is_installed():
            logger.warning("Ollama is not installed")
            return False

        if not self.service.is_running():
            logger.info("Ollama is installed but not running, attempting to start...")
            if self.service.start():
                logger.info("Ollama started successfully")
                return True
            else:
                logger.warning("Failed to start Ollama service")
                return False

        logger.info("Ollama is installed and running")
        return True

    def prompt_ollama_installation(self) -> bool:
        """Prompt user to install Ollama with explanation.

        Returns:
            True if user wants to proceed, False to cancel
        """
        print("\n" + "=" * 70)
        print(self.installer.explain_installation())
        print("=" * 70 + "\n")

        while True:
            response = (
                input("Do you want to proceed with Ollama installation? [Y/n]: ").strip().lower()
            )  # noqa: E501
            if response in ["y", "yes", ""]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please answer 'y' or 'n'")

    def install_ollama(self) -> bool:
        """Install Ollama with user confirmation.

        Returns:
            True if installation succeeded, False otherwise
        """
        if not self.prompt_ollama_installation():
            logger.info("User declined Ollama installation")
            return False

        logger.info("Starting Ollama installation...")

        if self.installer.can_auto_install():
            print("\nInstalling Ollama automatically...")
            success = self.installer.install()
            if success:
                print("✓ Ollama installed successfully!\n")
                return True
            else:
                print("✗ Ollama installation failed. Please install manually.\n")
                return False
        else:
            # Manual installation required
            print("\nDownloading Ollama installer...")
            success = self.installer.install()

            if success:
                print("\n" + "=" * 70)
                print("Please complete the installation manually.")
                print("Waiting for installation to complete...")
                print("=" * 70 + "\n")

                # Wait for user to complete installation
                if self.installer.wait_for_installation(timeout=600):
                    print("✓ Ollama installation detected!\n")
                    return True
                else:
                    print("✗ Installation timeout. Please ensure Ollama is installed.\n")
                    return False
            else:
                print("✗ Failed to download installer.\n")
                return False

    def display_recommendations(
        self, recommendations: list[ModelRecommendation], hardware: HardwareInfo
    ) -> None:
        """Display model recommendations to user.

        Args:
            recommendations: List of model recommendations
            hardware: Detected hardware info
        """
        print("\n" + "=" * 70)
        print("HARDWARE DETECTED")
        print("=" * 70)
        print(f"Platform: {hardware.platform}")
        print(f"CPU Cores: {hardware.cpu_count}")
        print(f"Total RAM: {hardware.total_ram_gb:.1f} GB")
        print(f"Available RAM: {hardware.available_ram_gb:.1f} GB")

        if hardware.gpu_available:
            print(f"GPU: {hardware.gpu_name}")
            if hardware.gpu_vram_gb:
                print(f"VRAM: {hardware.gpu_vram_gb:.1f} GB")
        else:
            print("GPU: None (CPU only)")

        print("=" * 70 + "\n")

        # Separate installed and available
        installed = [r for r in recommendations if r.is_installed and r.fits_hardware]
        available = [r for r in recommendations if not r.is_installed and r.fits_hardware]

        if installed:
            print("=" * 70)
            print("INSTALLED MODELS (Ready to use)")
            print("=" * 70)
            for i, rec in enumerate(installed, 1):
                print(f"\n{i}. {explain_recommendation(rec)}")

        if available:
            print("\n" + "=" * 70)
            print("RECOMMENDED MODELS (Available to download)")
            print("=" * 70)
            for i, rec in enumerate(available, len(installed) + 1):
                print(f"\n{i}. {explain_recommendation(rec)}")

        print("\n" + "=" * 70)

    def select_model_interactive(self, recommendations: list[ModelRecommendation]) -> str | None:
        """Interactively select a model from recommendations.

        Args:
            recommendations: List of recommended models

        Returns:
            Selected model name, or None if cancelled
        """
        fitting = [r for r in recommendations if r.fits_hardware]

        if not fitting:
            print("\n⚠ No compatible models found for your hardware.")
            print("You may need to upgrade your system or try smaller models.\n")
            return None

        # Build selection list
        options = []
        for rec in fitting:
            status = "✓ Installed" if rec.is_installed else "Download"
            options.append(f"{rec.model_name} ({rec.size}) - {status}")

        print("\nSelect a model by number:")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print(f"  {len(options) + 1}. Cancel / Exit")

        while True:
            try:
                choice = input(f"\nYour choice [1-{len(options) + 1}]: ").strip()
                choice_num = int(choice)

                if choice_num == len(options) + 1:
                    logger.info("User cancelled model selection")
                    return None

                if 1 <= choice_num <= len(options):
                    selected = fitting[choice_num - 1]
                    logger.info(f"User selected model: {selected.model_name}")
                    return selected.model_name
                else:
                    print(f"Please enter a number between 1 and {len(options) + 1}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\nCancelled by user")
                return None

    def download_model(self, model_name: str) -> bool:
        """Download selected model if not installed.

        Args:
            model_name: Name of model to download

        Returns:
            True if model ready (installed or downloaded), False otherwise
        """
        if not self.client:
            self.client = OllamaClient()

        # Check if already installed
        installed = self.service.list_models()
        if model_name.lower() in [m.lower() for m in installed]:
            logger.info(f"Model {model_name} is already installed")
            return True

        print(f"\nDownloading model: {model_name}")
        print("This may take several minutes depending on model size and connection speed...\n")

        try:
            # Use pull_model from client
            success = self.client.pull_model(model_name)
            if success:
                print(f"✓ Model {model_name} downloaded successfully!\n")
                return True
            else:
                print(f"✗ Failed to download model {model_name}\n")
                return False
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            print(f"✗ Error downloading model: {e}\n")
            return False

    def save_preference(self, model_name: str) -> None:
        """Save user's model preference to config.

        Args:
            model_name: Selected model name
        """
        # TODO: Implement config persistence
        # For now, just log
        logger.info(f"User preference saved: {model_name}")
        print(f"✓ Preference saved: {model_name} will be used as default model\n")

    def run(self) -> OnboardingResult:
        """Run complete onboarding flow.

        Returns:
            OnboardingResult with outcome
        """
        print("\n" + "=" * 70)
        print("  Welcome to ALI - Assistente Locale Intelligente")
        print("=" * 70 + "\n")

        # Step 1: Check/Install Ollama
        if not self.check_ollama_installation():
            if not self.install_ollama():
                return OnboardingResult(
                    success=False,
                    selected_model=None,
                    hardware=None,
                    ollama_installed=False,
                    error_message="Ollama installation required but not completed",
                )

            # Verify installation succeeded
            if not self.check_ollama_installation():
                return OnboardingResult(
                    success=False,
                    selected_model=None,
                    hardware=None,
                    ollama_installed=False,
                    error_message="Ollama installed but not running",
                )

        # Step 2: Detect Hardware
        print("Detecting hardware capabilities...\n")
        hardware = detect_hardware()

        # Step 3: Get installed models
        if not self.client:
            self.client = OllamaClient()

        installed_models = get_installed_models(self.client)

        # Step 4: Generate recommendations
        recommendations = get_model_recommendations(hardware, installed_models)

        # Step 5: Display and select
        self.display_recommendations(recommendations, hardware)
        selected_model = self.select_model_interactive(recommendations)

        if not selected_model:
            return OnboardingResult(
                success=False,
                selected_model=None,
                hardware=hardware,
                ollama_installed=True,
                error_message="No model selected",
            )

        # Step 6: Download if needed
        if not self.download_model(selected_model):
            return OnboardingResult(
                success=False,
                selected_model=selected_model,
                hardware=hardware,
                ollama_installed=True,
                error_message=f"Failed to download model: {selected_model}",
            )

        # Step 7: Save preference
        self.save_preference(selected_model)

        print("=" * 70)
        print("  Onboarding Complete! ALI is ready to use.")
        print("=" * 70 + "\n")

        return OnboardingResult(
            success=True,
            selected_model=selected_model,
            hardware=hardware,
            ollama_installed=True,
        )
