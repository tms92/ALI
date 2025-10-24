"""Automatic installation of required dependencies."""

import platform
import subprocess
from pathlib import Path

from loguru import logger

from ali.core.services import OllamaService


class OllamaInstaller:
    """Handle Ollama installation across platforms."""

    INSTALL_URLS = {
        "Windows": "https://ollama.com/download/OllamaSetup.exe",
        "Darwin": "https://ollama.com/download/Ollama-darwin.zip",
        "Linux": None,  # Linux uses curl install script
    }

    def __init__(self) -> None:
        """Initialize Ollama installer."""
        self.service = OllamaService()
        self.platform = platform.system()

    def is_installed(self) -> bool:
        """Check if Ollama is already installed.

        Returns:
            True if Ollama is installed, False otherwise
        """
        return self.service.is_installed()

    def get_install_command(self) -> str | None:
        """Get the installation command for current platform.

        Returns:
            Installation command string, or None if unsupported
        """
        if self.platform == "Linux":
            return "curl -fsSL https://ollama.com/install.sh | sh"
        elif self.platform == "Darwin":
            return "Download and run installer from https://ollama.com/download"
        elif self.platform == "Windows":
            return "Download and run OllamaSetup.exe from https://ollama.com/download"
        return None

    def explain_installation(self) -> str:
        """Generate user-friendly explanation of what will be installed.

        Returns:
            Formatted explanation string
        """
        explanation = """
Ollama Installation Required
=============================

ALI needs Ollama to run local AI models on your computer.

What is Ollama?
- Free, open-source tool for running AI models locally
- Runs models like Llama, Mistral, Gemma on your hardware
- No cloud, no API keys, complete privacy
- Official website: https://ollama.com

What will happen:
"""
        if self.platform == "Linux":
            explanation += """
1. Download official Ollama installer script
2. Install Ollama binary to /usr/local/bin
3. Set up Ollama as a system service
4. Start the Ollama service automatically

Command: curl -fsSL https://ollama.com/install.sh | sh
"""
        elif self.platform == "Darwin":
            explanation += """
1. Download Ollama installer for macOS
2. You'll need to manually run the installer
3. Ollama will be installed to /Applications
4. The service will start automatically

Download from: https://ollama.com/download
"""
        elif self.platform == "Windows":
            explanation += """
1. Download Ollama installer for Windows
2. You'll need to manually run OllamaSetup.exe
3. Ollama will be installed and added to PATH
4. The service will start automatically

Download from: https://ollama.com/download
"""
        else:
            explanation += f"\nUnsupported platform: {self.platform}\n"

        explanation += """
After installation, ALI will automatically:
- Detect your hardware (RAM, GPU)
- Recommend compatible AI models
- Help you download and configure your first model

Size: ~500MB download, ~1GB installed
Time: 2-5 minutes depending on your connection
"""
        return explanation.strip()

    def can_auto_install(self) -> bool:
        """Check if automatic installation is supported on this platform.

        Returns:
            True if can auto-install, False if manual installation required
        """
        # Only Linux supports fully automated installation
        return self.platform == "Linux"

    def install_linux(self) -> bool:
        """Install Ollama on Linux using official script.

        Returns:
            True if installation succeeded, False otherwise
        """
        logger.info("Starting Ollama installation on Linux")

        try:
            # Download and execute install script
            cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            logger.debug(f"Running: {cmd}")

            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if process.returncode == 0:
                logger.info("Ollama installed successfully")
                return True
            else:
                logger.error(f"Installation failed: {process.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Installation timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False

    def download_installer_windows(self, dest_path: Path) -> bool:
        """Download Windows installer to specified path.

        Args:
            dest_path: Where to save the installer

        Returns:
            True if download succeeded, False otherwise
        """
        import requests  # type: ignore[import-untyped]

        url = self.INSTALL_URLS["Windows"]
        if not url:
            return False

        try:
            logger.info(f"Downloading Windows installer from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Installer downloaded to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def get_download_path_windows(self) -> Path:
        """Get path where Windows installer should be downloaded.

        Returns:
            Path to download location
        """
        # Use user's Downloads folder
        downloads = Path.home() / "Downloads"
        return downloads / "OllamaSetup.exe"

    def install(self) -> bool:
        """Install Ollama on current platform.

        Returns:
            True if installation succeeded or user needs to complete manually
        """
        if self.is_installed():
            logger.info("Ollama is already installed, skipping installation")
            return True

        if self.platform == "Linux":
            return self.install_linux()

        elif self.platform == "Windows":
            dest = self.get_download_path_windows()
            if self.download_installer_windows(dest):
                logger.info(f"Installer downloaded to {dest}")
                logger.info("Please run the installer to complete setup")
                # Try to open the installer
                try:
                    subprocess.Popen([str(dest)], shell=True)
                except Exception as e:
                    logger.warning(f"Could not auto-open installer: {e}")
                return True
            return False

        elif self.platform == "Darwin":
            logger.info("Please download installer from https://ollama.com/download")
            logger.info("ALI will wait for you to complete the installation")
            return True

        else:
            logger.error(f"Unsupported platform: {self.platform}")
            return False

    def wait_for_installation(self, timeout: int = 600, check_interval: int = 5) -> bool:
        """Wait for user to complete manual installation.

        Args:
            timeout: Maximum time to wait in seconds (default 10 minutes)
            check_interval: How often to check in seconds (default 5 seconds)

        Returns:
            True if Ollama becomes available, False if timeout
        """
        import time

        logger.info("Waiting for Ollama installation to complete...")
        elapsed = 0

        while elapsed < timeout:
            if self.is_installed():
                logger.info("Ollama installation detected!")
                return True

            time.sleep(check_interval)
            elapsed += check_interval

            if elapsed % 30 == 0:  # Log every 30 seconds
                logger.debug(f"Still waiting... ({elapsed}s / {timeout}s)")

        logger.warning(f"Timeout waiting for installation after {timeout}s")
        return False
