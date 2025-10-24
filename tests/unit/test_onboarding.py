"""Tests for onboarding flow."""

from unittest.mock import Mock, patch

import pytest

from ali.core.hardware import HardwareInfo
from ali.core.onboarding import OnboardingFlow, OnboardingResult
from ali.core.recommendations import ModelRecommendation


@pytest.fixture
def hardware_mock() -> HardwareInfo:
    """Mock hardware info."""
    return HardwareInfo(
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        cpu_count=8,
        gpu_available=False,
        gpu_name=None,
        gpu_vram_gb=None,
        platform="Linux",
    )


@pytest.fixture
def recommendations_mock() -> list[ModelRecommendation]:
    """Mock model recommendations."""
    return [
        ModelRecommendation(
            model_name="llama3.2:3b",
            size="3B",
            reason="Great fit",
            ram_required_gb=4.0,
            fits_hardware=True,
            performance_tier="optimal",
            is_installed=True,
            is_outdated=False,
        ),
        ModelRecommendation(
            model_name="llama3.1:8b",
            size="8B",
            reason="Good fit",
            ram_required_gb=8.0,
            fits_hardware=True,
            performance_tier="good",
            is_installed=False,
            is_outdated=False,
        ),
    ]


@pytest.fixture
def onboarding() -> OnboardingFlow:
    """Create onboarding flow instance."""
    with patch("ali.core.onboarding.OllamaService"):
        with patch("ali.core.onboarding.OllamaInstaller"):
            return OnboardingFlow()


def test_should_initialize_onboarding(onboarding: OnboardingFlow) -> None:
    """Test onboarding initialization."""
    # Assert
    assert onboarding.service is not None
    assert onboarding.installer is not None
    assert onboarding.client is None  # Not initialized yet


def test_should_detect_ollama_installed_and_running() -> None:
    """Test Ollama detection when installed and running."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService") as mock_service:
        with patch("ali.core.onboarding.OllamaInstaller"):
            mock_service.return_value.is_installed.return_value = True
            mock_service.return_value.is_running.return_value = True

            onboarding = OnboardingFlow()

            # Act
            result = onboarding.check_ollama_installation()

            # Assert
            assert result is True
            mock_service.return_value.is_installed.assert_called_once()
            mock_service.return_value.is_running.assert_called_once()


def test_should_detect_ollama_not_installed() -> None:
    """Test Ollama detection when not installed."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService") as mock_service:
        with patch("ali.core.onboarding.OllamaInstaller"):
            mock_service.return_value.is_installed.return_value = False

            onboarding = OnboardingFlow()

            # Act
            result = onboarding.check_ollama_installation()

            # Assert
            assert result is False
            mock_service.return_value.is_installed.assert_called_once()


def test_should_start_ollama_if_not_running() -> None:
    """Test starting Ollama if installed but not running."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService") as mock_service:
        with patch("ali.core.onboarding.OllamaInstaller"):
            mock_service.return_value.is_installed.return_value = True
            mock_service.return_value.is_running.return_value = False
            mock_service.return_value.start.return_value = True

            onboarding = OnboardingFlow()

            # Act
            result = onboarding.check_ollama_installation()

            # Assert
            assert result is True
            mock_service.return_value.start.assert_called_once()


@patch("builtins.input", return_value="y")
def test_should_accept_installation_prompt(mock_input: Mock) -> None:
    """Test user accepts installation."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService"):
        with patch("ali.core.onboarding.OllamaInstaller") as mock_installer:
            mock_installer.return_value.explain_installation.return_value = "Explanation"
            onboarding = OnboardingFlow()

            # Act
            with patch("builtins.print"):  # Suppress output
                result = onboarding.prompt_ollama_installation()

            # Assert
            assert result is True


@patch("builtins.input", return_value="n")
def test_should_decline_installation_prompt(mock_input: Mock) -> None:
    """Test user declines installation."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService"):
        with patch("ali.core.onboarding.OllamaInstaller") as mock_installer:
            mock_installer.return_value.explain_installation.return_value = "Explanation"
            onboarding = OnboardingFlow()

            # Act
            with patch("builtins.print"):
                result = onboarding.prompt_ollama_installation()

            # Assert
            assert result is False


@patch("builtins.input", side_effect=["invalid", "y"])
def test_should_retry_on_invalid_input(mock_input: Mock) -> None:
    """Test retry on invalid input."""
    # Arrange
    with patch("ali.core.onboarding.OllamaService"):
        with patch("ali.core.onboarding.OllamaInstaller") as mock_installer:
            mock_installer.return_value.explain_installation.return_value = "Explanation"
            onboarding = OnboardingFlow()

            # Act
            with patch("builtins.print"):
                result = onboarding.prompt_ollama_installation()

            # Assert
            assert result is True
            assert mock_input.call_count == 2


def test_should_display_recommendations(
    onboarding: OnboardingFlow,
    hardware_mock: HardwareInfo,
    recommendations_mock: list[ModelRecommendation],  # noqa: E501
) -> None:
    """Test displaying recommendations."""
    # Act & Assert - just verify it doesn't crash
    with patch("builtins.print"):
        onboarding.display_recommendations(recommendations_mock, hardware_mock)


@patch("builtins.input", return_value="1")
def test_should_select_first_model(
    mock_input: Mock,
    onboarding: OnboardingFlow,
    recommendations_mock: list[ModelRecommendation],  # noqa: E501
) -> None:
    """Test selecting first model."""
    # Act
    with patch("builtins.print"):
        result = onboarding.select_model_interactive(recommendations_mock)

    # Assert
    assert result == "llama3.2:3b"


@patch("builtins.input", return_value="2")
def test_should_select_second_model(
    mock_input: Mock,
    onboarding: OnboardingFlow,
    recommendations_mock: list[ModelRecommendation],  # noqa: E501
) -> None:
    """Test selecting second model."""
    # Act
    with patch("builtins.print"):
        result = onboarding.select_model_interactive(recommendations_mock)

    # Assert
    assert result == "llama3.1:8b"


@patch("builtins.input", return_value="3")
def test_should_cancel_selection(
    mock_input: Mock,
    onboarding: OnboardingFlow,
    recommendations_mock: list[ModelRecommendation],  # noqa: E501
) -> None:
    """Test cancelling selection."""
    # Act
    with patch("builtins.print"):
        result = onboarding.select_model_interactive(recommendations_mock)

    # Assert
    assert result is None


@patch("builtins.input", side_effect=["invalid", "999", "1"])
def test_should_handle_invalid_selection(
    mock_input: Mock,
    onboarding: OnboardingFlow,
    recommendations_mock: list[ModelRecommendation],  # noqa: E501
) -> None:
    """Test handling invalid selections."""
    # Act
    with patch("builtins.print"):
        result = onboarding.select_model_interactive(recommendations_mock)

    # Assert
    assert result == "llama3.2:3b"
    assert mock_input.call_count == 3


def test_should_return_none_for_no_compatible_models(onboarding: OnboardingFlow) -> None:
    """Test selection with no compatible models."""
    # Arrange
    no_fit = [
        ModelRecommendation(
            model_name="llama3.1:70b",
            size="70B",
            reason="Too large",
            ram_required_gb=64.0,
            fits_hardware=False,
            performance_tier="minimal",
            is_installed=False,
            is_outdated=False,
        )
    ]

    # Act
    with patch("builtins.print"):
        result = onboarding.select_model_interactive(no_fit)

    # Assert
    assert result is None


def test_should_skip_download_if_installed(onboarding: OnboardingFlow) -> None:
    """Test skipping download for installed model."""
    # Arrange
    with patch("ali.core.onboarding.OllamaClient"):
        onboarding.service.list_models = Mock(return_value=["llama3.2:3b"])

        # Act
        with patch("builtins.print"):
            result = onboarding.download_model("llama3.2:3b")

        # Assert
        assert result is True


def test_should_download_model_if_not_installed(onboarding: OnboardingFlow) -> None:
    """Test downloading model."""
    # Arrange
    with patch("ali.core.onboarding.OllamaClient") as mock_client:
        mock_client.return_value.pull_model.return_value = True
        onboarding.service.list_models = Mock(return_value=[])
        onboarding.client = mock_client.return_value

        # Act
        with patch("builtins.print"):
            result = onboarding.download_model("llama3.1:8b")

        # Assert
        assert result is True
        mock_client.return_value.pull_model.assert_called_once_with("llama3.1:8b")


def test_should_handle_download_failure(onboarding: OnboardingFlow) -> None:
    """Test handling download failure."""
    # Arrange
    with patch("ali.core.onboarding.OllamaClient") as mock_client:
        mock_client.return_value.pull_model.return_value = False
        onboarding.service.list_models = Mock(return_value=[])
        onboarding.client = mock_client.return_value

        # Act
        with patch("builtins.print"):
            result = onboarding.download_model("llama3.1:8b")

        # Assert
        assert result is False


def test_should_save_preference(onboarding: OnboardingFlow) -> None:
    """Test saving user preference."""
    # Act & Assert - just verify it doesn't crash
    with patch("builtins.print"):
        onboarding.save_preference("llama3.2:3b")


def test_onboarding_result_dataclass() -> None:
    """Test OnboardingResult dataclass."""
    # Arrange & Act
    result = OnboardingResult(
        success=True,
        selected_model="llama3.2:3b",
        hardware=None,
        ollama_installed=True,
    )

    # Assert
    assert result.success is True
    assert result.selected_model == "llama3.2:3b"
    assert result.ollama_installed is True
    assert result.error_message is None
