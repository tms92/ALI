"""Tests for model recommendation system."""

import pytest

from ali.core.hardware import HardwareInfo
from ali.core.recommendations import (
    AVAILABLE_MODELS,
    ModelRecommendation,
    detect_obsolete_models,
    explain_recommendation,
    get_model_recommendations,
    get_top_recommendations,
)


@pytest.fixture
def hardware_cpu_only() -> HardwareInfo:
    """Hardware with CPU only, 16GB RAM."""
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
def hardware_with_gpu() -> HardwareInfo:
    """Hardware with GPU, 32GB RAM, 12GB VRAM."""
    return HardwareInfo(
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        cpu_count=16,
        gpu_available=True,
        gpu_name="NVIDIA RTX 3080",
        gpu_vram_gb=12.0,
        platform="Linux",
    )


@pytest.fixture
def hardware_low_spec() -> HardwareInfo:
    """Low-spec hardware with 4GB RAM."""
    return HardwareInfo(
        total_ram_gb=4.0,
        available_ram_gb=3.0,
        cpu_count=2,
        gpu_available=False,
        gpu_name=None,
        gpu_vram_gb=None,
        platform="Windows",
    )


def test_get_model_recommendations_cpu_only(hardware_cpu_only: HardwareInfo) -> None:
    """Test model recommendations for CPU-only system."""
    # Arrange - no models installed
    installed_models: list[str] = []

    # Act
    recommendations = get_model_recommendations(hardware_cpu_only, installed_models)

    # Assert
    assert len(recommendations) > 0
    # All recommendations should have is_installed=False
    assert all(not rec.is_installed for rec in recommendations)
    # Should have models that fit
    fitting = [r for r in recommendations if r.fits_hardware]
    assert len(fitting) > 0
    # Fitting models should be sorted first
    assert recommendations[0].fits_hardware


def test_get_model_recommendations_with_installed(hardware_cpu_only: HardwareInfo) -> None:
    """Test model recommendations with some models already installed."""
    # Arrange
    installed_models = ["llama3.2:3b", "llama3.1:8b"]

    # Act
    recommendations = get_model_recommendations(hardware_cpu_only, installed_models)

    # Assert
    installed_recs = [r for r in recommendations if r.is_installed]
    assert len(installed_recs) == 2
    # Installed models should be sorted first (among fitting models)
    fitting = [r for r in recommendations if r.fits_hardware]
    if len(fitting) >= 2:
        assert fitting[0].is_installed or fitting[1].is_installed


def test_get_model_recommendations_gpu(hardware_with_gpu: HardwareInfo) -> None:
    """Test model recommendations for GPU system."""
    # Arrange
    installed_models: list[str] = []

    # Act
    recommendations = get_model_recommendations(hardware_with_gpu, installed_models)

    # Assert
    assert len(recommendations) > 0
    # Should have optimal tier recommendations with GPU
    optimal = [r for r in recommendations if r.performance_tier == "optimal"]
    assert len(optimal) > 0
    # GPU should be mentioned in reasons
    fitting = [r for r in recommendations if r.fits_hardware]
    assert any("GPU" in r.reason for r in fitting)


def test_get_model_recommendations_low_spec(hardware_low_spec: HardwareInfo) -> None:
    """Test model recommendations for low-spec system."""
    # Arrange
    installed_models: list[str] = []

    # Act
    recommendations = get_model_recommendations(hardware_low_spec, installed_models)

    # Assert
    # Should still have recommendations
    assert len(recommendations) > 0
    # Should have at least small models that fit
    fitting = [r for r in recommendations if r.fits_hardware]
    assert len(fitting) > 0
    # Smallest models should fit
    smallest = [r for r in recommendations if r.size == "1B"]
    assert len(smallest) > 0
    assert smallest[0].fits_hardware


def test_get_top_recommendations(hardware_cpu_only: HardwareInfo) -> None:
    """Test getting top N recommendations."""
    # Arrange
    installed_models = ["llama3.2:3b"]

    # Act
    top_3 = get_top_recommendations(hardware_cpu_only, installed_models, count=3)

    # Assert
    assert len(top_3) <= 3
    assert all(rec.fits_hardware for rec in top_3)
    # Installed model should be first if it fits
    if top_3[0].model_name == "llama3.2:3b":
        assert top_3[0].is_installed


def test_detect_obsolete_models() -> None:
    """Test detection of obsolete models."""
    # Arrange
    installed_models = ["llama3:7b", "llama2:13b", "mistral:7b"]

    # Act
    obsolete = detect_obsolete_models(installed_models, AVAILABLE_MODELS)

    # Assert
    # llama3:7b is obsolete (we have llama3.1:8b and llama3.2:8b)
    # llama2:13b is obsolete (we have llama3.x variants)
    # mistral:7b is current (it's in our list)
    assert "llama3:7b" in obsolete or "llama2:13b" in obsolete
    assert "mistral:7b" not in obsolete  # This one is current


def test_detect_obsolete_models_empty() -> None:
    """Test obsolete detection with no installed models."""
    # Arrange
    installed_models: list[str] = []

    # Act
    obsolete = detect_obsolete_models(installed_models)

    # Assert
    assert len(obsolete) == 0


def test_explain_recommendation_installed() -> None:
    """Test explanation for installed model."""
    # Arrange
    rec = ModelRecommendation(
        model_name="llama3.2:3b",
        size="3B",
        reason="Great fit for your RAM",
        ram_required_gb=4.0,
        fits_hardware=True,
        performance_tier="optimal",
        is_installed=True,
        is_outdated=False,
    )

    # Act
    explanation = explain_recommendation(rec)

    # Assert
    assert "llama3.2:3b" in explanation
    assert "3B" in explanation
    assert "✓ INSTALLED" in explanation
    assert "OPTIMAL" in explanation
    assert "4.0GB" in explanation


def test_explain_recommendation_not_installed() -> None:
    """Test explanation for model not installed."""
    # Arrange
    rec = ModelRecommendation(
        model_name="llama3.1:8b",
        size="8B",
        reason="Will run well on your GPU",
        ram_required_gb=6.0,
        fits_hardware=True,
        performance_tier="good",
        is_installed=False,
        is_outdated=False,
    )

    # Act
    explanation = explain_recommendation(rec)

    # Assert
    assert "llama3.1:8b" in explanation
    assert "Available to download" in explanation
    assert "GOOD" in explanation


def test_explain_recommendation_outdated() -> None:
    """Test explanation for outdated model."""
    # Arrange
    rec = ModelRecommendation(
        model_name="llama2:7b",
        size="7B",
        reason="Installed but outdated",
        ram_required_gb=8.0,
        fits_hardware=True,
        performance_tier="good",
        is_installed=True,
        is_outdated=True,
    )

    # Act
    explanation = explain_recommendation(rec)

    # Assert
    assert "llama2:7b" in explanation
    assert "⚠ OUTDATED" in explanation
    assert "Update available" in explanation


def test_model_recommendation_dataclass() -> None:
    """Test ModelRecommendation dataclass."""
    # Arrange & Act
    rec = ModelRecommendation(
        model_name="test:model",
        size="7B",
        reason="Test reason",
        ram_required_gb=8.0,
        fits_hardware=True,
        performance_tier="optimal",
        is_installed=False,
        is_outdated=False,
    )

    # Assert
    assert rec.model_name == "test:model"
    assert rec.size == "7B"
    assert rec.reason == "Test reason"
    assert rec.ram_required_gb == 8.0
    assert rec.fits_hardware is True
    assert rec.performance_tier == "optimal"
    assert rec.is_installed is False
    assert rec.is_outdated is False


def test_case_insensitive_matching(hardware_cpu_only: HardwareInfo) -> None:
    """Test that model matching is case-insensitive."""
    # Arrange
    installed_models = ["LLAMA3.2:3B", "Llama3.1:8b"]

    # Act
    recommendations = get_model_recommendations(hardware_cpu_only, installed_models)

    # Assert
    installed_recs = [r for r in recommendations if r.is_installed]
    # Should match despite different case
    assert len(installed_recs) == 2


def test_get_model_recommendations_none_installed(hardware_cpu_only: HardwareInfo) -> None:
    """Test model recommendations when installed_models is None."""
    # Arrange - explicitly pass None
    installed_models = None

    # Act
    recommendations = get_model_recommendations(hardware_cpu_only, installed_models)

    # Assert
    assert len(recommendations) > 0
    # No models should be marked as installed
    assert all(not rec.is_installed for rec in recommendations)


def test_installed_model_too_large_for_hardware(hardware_low_spec: HardwareInfo) -> None:
    """Test when user has installed a model that's too large for their hardware."""
    # Arrange - user installed a 70B model on low-spec hardware (3GB RAM)
    installed_models = ["llama3.1:70b"]

    # Act
    recommendations = get_model_recommendations(hardware_low_spec, installed_models)

    # Assert
    # Find the 70B model recommendation
    large_model = next((r for r in recommendations if r.model_name == "llama3.1:70b"), None)
    assert large_model is not None
    assert large_model.is_installed is True
    assert large_model.fits_hardware is False
    # Should have warning about struggling
    assert "⚠ Installed but may struggle" in large_model.reason


def test_get_installed_models_success() -> None:
    """Test successful retrieval of installed models."""
    # Arrange
    from unittest.mock import Mock

    from ali.core.recommendations import get_installed_models

    mock_client = Mock()
    mock_client.list_models.return_value = ["llama3.2:3b", "mistral:7b", "gemma2:9b"]

    # Act
    result = get_installed_models(mock_client)

    # Assert
    assert len(result) == 3
    assert "llama3.2:3b" in result
    assert "mistral:7b" in result
    assert "gemma2:9b" in result
    mock_client.list_models.assert_called_once()


def test_get_installed_models_error() -> None:
    """Test get_installed_models when client raises exception."""
    # Arrange
    from unittest.mock import Mock

    from ali.core.recommendations import get_installed_models

    mock_client = Mock()
    mock_client.list_models.side_effect = Exception("Connection failed")

    # Act
    result = get_installed_models(mock_client)

    # Assert
    # Should return empty list on error
    assert result == []
    mock_client.list_models.assert_called_once()
