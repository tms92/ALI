"""Tests for Ollama installer."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from ali.core.installer import OllamaInstaller


@pytest.fixture
def mock_service() -> Mock:
    """Create mock OllamaService."""
    with patch("ali.core.installer.OllamaService") as mock:
        yield mock


@pytest.fixture
def installer(mock_service: Mock) -> OllamaInstaller:
    """Create OllamaInstaller instance with mocked service."""
    mock_service.return_value.is_installed.return_value = False
    return OllamaInstaller()


def test_should_initialize_installer(installer: OllamaInstaller) -> None:
    """Test installer initialization."""
    # Assert
    assert installer.service is not None
    assert installer.platform in ["Windows", "Darwin", "Linux"]


@pytest.mark.parametrize("is_installed", [True, False])
def test_should_detect_installation_status(mock_service: Mock, is_installed: bool) -> None:
    """Test detection of Ollama installation status."""
    # Arrange
    mock_service.return_value.is_installed.return_value = is_installed
    installer = OllamaInstaller()

    # Act
    result = installer.is_installed()

    # Assert
    assert result is is_installed
    mock_service.return_value.is_installed.assert_called_once()


@pytest.mark.parametrize(
    "platform,expected_content",
    [
        ("Linux", "curl"),
        ("Linux", "ollama.com/install.sh"),
        ("Darwin", "ollama.com/download"),
        ("Windows", "OllamaSetup.exe"),
        ("Windows", "ollama.com/download"),
    ],
)
def test_should_return_correct_install_command(
    mock_service: Mock, platform: str, expected_content: str
) -> None:
    """Test install command for different platforms."""
    # Arrange
    with patch("ali.core.installer.platform.system", return_value=platform):
        installer = OllamaInstaller()

        # Act
        command = installer.get_install_command()

        # Assert
        assert command is not None
        assert expected_content in command


@pytest.mark.parametrize(
    "platform,expected_keywords",
    [
        ("Linux", ["Ollama", "curl", "ollama.com", "What will happen"]),
        ("Darwin", ["Ollama", "macOS", "ollama.com/download", "What will happen"]),
        ("Windows", ["Ollama", "Windows", "OllamaSetup.exe", "What will happen"]),
    ],
)
def test_should_explain_installation(
    mock_service: Mock, platform: str, expected_keywords: list[str]
) -> None:
    """Test installation explanation contains required information."""
    # Arrange
    with patch("ali.core.installer.platform.system", return_value=platform):
        installer = OllamaInstaller()

        # Act
        explanation = installer.explain_installation()

        # Assert
        for keyword in expected_keywords:
            assert keyword in explanation


@pytest.mark.parametrize(
    "platform,can_auto_install",
    [
        ("Linux", True),
        ("Darwin", False),
        ("Windows", False),
    ],
)
def test_should_report_auto_install_capability(
    mock_service: Mock, platform: str, can_auto_install: bool
) -> None:
    """Test auto-install capability for different platforms."""
    # Arrange
    with patch("ali.core.installer.platform.system", return_value=platform):
        installer = OllamaInstaller()

        # Act
        result = installer.can_auto_install()

        # Assert
        assert result is can_auto_install


def test_should_skip_install_if_already_installed(mock_service: Mock) -> None:
    """Test that install() skips if Ollama is already installed."""
    # Arrange
    mock_service.return_value.is_installed.return_value = True
    installer = OllamaInstaller()

    # Act
    result = installer.install()

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "returncode,expected_result",
    [
        (0, True),  # Success
        (1, False),  # Failure
    ],
)
def test_should_handle_linux_installation(
    mock_service: Mock, returncode: int, expected_result: bool
) -> None:
    """Test Linux installation with different outcomes."""
    # Arrange
    with patch("ali.core.installer.subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=returncode, stderr="Error message")
        with patch("ali.core.installer.platform.system", return_value="Linux"):
            installer = OllamaInstaller()

            # Act
            result = installer.install_linux()

            # Assert
            assert result is expected_result
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "curl" in call_args[0][0]
            assert "ollama.com/install.sh" in call_args[0][0]


def test_should_handle_linux_install_timeout(mock_service: Mock) -> None:
    """Test Linux installation timeout."""
    # Arrange
    import subprocess

    with patch("ali.core.installer.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)
        with patch("ali.core.installer.platform.system", return_value="Linux"):
            installer = OllamaInstaller()

            # Act
            result = installer.install_linux()

            # Assert
            assert result is False


def test_should_handle_linux_install_exception(mock_service: Mock) -> None:
    """Test Linux installation with unexpected exception."""
    # Arrange
    with patch("ali.core.installer.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Unexpected error")
        with patch("ali.core.installer.platform.system", return_value="Linux"):
            installer = OllamaInstaller()

            # Act
            result = installer.install_linux()

            # Assert
            assert result is False


def test_should_get_windows_download_path(mock_service: Mock) -> None:
    """Test Windows download path."""
    # Arrange
    with patch("ali.core.installer.platform.system", return_value="Windows"):
        installer = OllamaInstaller()

        # Act
        path = installer.get_download_path_windows()

        # Assert
        assert isinstance(path, Path)
        assert path.name == "OllamaSetup.exe"
        assert "Downloads" in str(path)


def test_should_download_windows_installer(mock_service: Mock) -> None:
    """Test Windows installer download."""
    # Arrange
    import requests

    mock_response = Mock()
    mock_response.iter_content.return_value = [b"fake", b"data"]

    with patch.object(requests, "get", return_value=mock_response):
        with patch("ali.core.installer.platform.system", return_value="Windows"):
            with patch("builtins.open", mock_open()) as mock_file:
                installer = OllamaInstaller()
                dest = Path("/fake/path/OllamaSetup.exe")

                # Act
                result = installer.download_installer_windows(dest)

                # Assert
                assert result is True
                mock_file.assert_called_once_with(dest, "wb")


def test_should_handle_windows_download_failure(mock_service: Mock) -> None:
    """Test Windows download failure."""
    # Arrange
    import requests

    with patch.object(requests, "get", side_effect=Exception("Network error")):
        with patch("ali.core.installer.platform.system", return_value="Windows"):
            installer = OllamaInstaller()
            dest = Path("/fake/path/installer.exe")

            # Act
            result = installer.download_installer_windows(dest)

            # Assert
            assert result is False


def test_should_wait_for_installation(mock_service: Mock) -> None:
    """Test waiting for manual installation."""
    # Arrange
    import time

    call_count = 0

    def check_installed() -> bool:
        nonlocal call_count
        call_count += 1
        # Become installed after 3 checks
        return call_count >= 3

    mock_service.return_value.is_installed.side_effect = check_installed
    installer = OllamaInstaller()

    # Act
    with patch.object(time, "sleep"):  # Speed up test
        result = installer.wait_for_installation(timeout=30, check_interval=1)

    # Assert
    assert result is True
    assert call_count >= 3


def test_should_timeout_waiting_for_installation(mock_service: Mock) -> None:
    """Test timeout when waiting for installation."""
    # Arrange
    import time

    mock_service.return_value.is_installed.return_value = False
    installer = OllamaInstaller()

    # Act
    with patch.object(time, "sleep"):  # Speed up test
        result = installer.wait_for_installation(timeout=5, check_interval=1)

    # Assert
    assert result is False
    # Should have checked multiple times before timeout
    assert mock_service.return_value.is_installed.call_count >= 5


def test_should_have_install_urls() -> None:
    """Test that install URLs are defined."""
    # Assert
    assert OllamaInstaller.INSTALL_URLS["Windows"] is not None
    assert "ollama.com" in OllamaInstaller.INSTALL_URLS["Windows"]
    assert OllamaInstaller.INSTALL_URLS["Darwin"] is not None
    assert OllamaInstaller.INSTALL_URLS["Linux"] is None  # Uses script
