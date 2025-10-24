"""Model recommendations based on hardware capabilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from ali.core.hardware import HardwareInfo

if TYPE_CHECKING:
    from ali.llm.client import OllamaClient


@dataclass
class ModelRecommendation:
    """Model recommendation with metadata."""

    model_name: str
    size: str  # e.g., "7B", "13B", "70B"
    reason: str
    ram_required_gb: float
    fits_hardware: bool
    performance_tier: str  # "optimal", "good", "minimal"
    is_installed: bool
    is_outdated: bool  # True if a newer version exists


# Model database with requirements
# Format: (model_name, size, ram_required_gb, vram_required_gb_if_gpu)
AVAILABLE_MODELS = [
    ("llama3.2:1b", "1B", 2.0, 1.5),
    ("llama3.2:3b", "3B", 4.0, 3.0),
    ("phi3:3b", "3B", 4.0, 3.0),
    ("llama3.2:8b", "8B", 8.0, 6.0),
    ("llama3.1:8b", "8B", 8.0, 6.0),
    ("gemma2:9b", "9B", 10.0, 7.0),
    ("mistral:7b", "7B", 8.0, 6.0),
    ("llama3.1:13b", "13B", 16.0, 10.0),
    ("llama3.1:70b", "70B", 64.0, 40.0),
    ("codellama:13b", "13B", 16.0, 10.0),
    ("codellama:34b", "34B", 32.0, 20.0),
]


def get_model_recommendations(
    hardware: HardwareInfo, installed_models: list[str] | None = None
) -> list[ModelRecommendation]:
    """Get model recommendations based on hardware and installed models.

    Args:
        hardware: Detected hardware information
        installed_models: List of installed model names (e.g., ["llama3.2:3b", "llama3.1:8b"])
                         If None, no models are considered installed

    Returns:
        List of recommended models, sorted by performance tier
    """
    logger.info("Generating model recommendations based on hardware")

    if installed_models is None:
        installed_models = []

    # Normalize installed model names for comparison
    installed_set = {model.lower() for model in installed_models}

    recommendations = []
    available_ram = hardware.available_ram_gb

    for model_name, size, ram_req, vram_req in AVAILABLE_MODELS:
        is_installed = model_name.lower() in installed_set

        # Determine if model fits
        if hardware.gpu_available and hardware.gpu_vram_gb:
            # GPU available - check VRAM
            fits = hardware.gpu_vram_gb >= vram_req
            ram_to_use = vram_req if fits else ram_req
        else:
            # CPU only - check RAM
            fits = available_ram >= ram_req
            ram_to_use = ram_req

        # Determine performance tier
        if fits:
            if hardware.gpu_available:
                if hardware.gpu_vram_gb and hardware.gpu_vram_gb >= vram_req * 1.5:
                    tier = "optimal"
                    reason = f"Excellent fit for your GPU ({hardware.gpu_name})"
                else:
                    tier = "good"
                    reason = f"Will run well on your GPU ({hardware.gpu_name})"
            else:
                if available_ram >= ram_req * 2:
                    tier = "optimal"
                    reason = f"Great fit for your RAM ({available_ram:.1f}GB available)"
                else:
                    tier = "good"
                    reason = f"Will run on CPU ({available_ram:.1f}GB RAM)"

            # Add installation status to reason
            if is_installed:
                reason = f"✓ Already installed - {reason}"
        else:
            tier = "minimal"
            if hardware.gpu_available:
                reason = f"Requires {vram_req:.1f}GB VRAM (you have {hardware.gpu_vram_gb:.1f}GB)"
            else:
                reason = f"Requires {ram_req:.1f}GB RAM (you have {available_ram:.1f}GB)"

            if is_installed:
                reason = f"⚠ Installed but may struggle - {reason}"

        rec = ModelRecommendation(
            model_name=model_name,
            size=size,
            reason=reason,
            ram_required_gb=ram_to_use,
            fits_hardware=fits,
            performance_tier=tier,
            is_installed=is_installed,
            is_outdated=False,  # Will be set later when checking for updates
        )
        recommendations.append(rec)

    # Sort: installed and fitting first, then fitting, then by tier, then by size
    tier_order = {"optimal": 0, "good": 1, "minimal": 2}
    recommendations.sort(
        key=lambda r: (
            not r.fits_hardware,  # fitting first
            not r.is_installed,  # installed models first within fitting group
            tier_order.get(r.performance_tier, 3),
            -r.ram_required_gb,  # larger models first within tier
        )
    )

    logger.info(
        f"Generated {len([r for r in recommendations if r.fits_hardware])} "
        f"compatible recommendations "
        f"({len([r for r in recommendations if r.is_installed])} installed)"
    )

    return recommendations


def get_top_recommendations(
    hardware: HardwareInfo, installed_models: list[str] | None = None, count: int = 3
) -> list[ModelRecommendation]:
    """Get top N model recommendations.

    Args:
        hardware: Detected hardware information
        installed_models: List of installed model names
        count: Number of recommendations to return

    Returns:
        List of top recommended models
    """
    all_recs = get_model_recommendations(hardware, installed_models)
    # Only return models that fit
    fitting = [r for r in all_recs if r.fits_hardware]
    return fitting[:count]


def get_installed_models(client: "OllamaClient") -> list[str]:
    """Get list of installed model names from Ollama.

    Args:
        client: Ollama client instance

    Returns:
        List of installed model names
    """
    try:
        model_names = client.list_models()
        logger.info(f"Found {len(model_names)} installed models")
        return model_names
    except Exception as e:
        logger.warning(f"Failed to get installed models: {e}")
        return []


def detect_obsolete_models(
    installed_models: list[str], available_models: list[tuple[str, str, float, float]] | None = None
) -> list[str]:
    """Detect installed models that have newer versions available.

    Args:
        installed_models: List of installed model names
        available_models: List of available models (defaults to AVAILABLE_MODELS)

    Returns:
        List of obsolete model names
    """
    if available_models is None:
        available_models = AVAILABLE_MODELS

    obsolete = []
    available_set = {model[0].lower() for model in available_models}

    for installed in installed_models:
        installed_lower = installed.lower()
        # Check if this exact model is not in the available list
        if installed_lower not in available_set:
            # Check if there's a similar model (same base name, different version)
            base_name = installed_lower.split(":")[0] if ":" in installed_lower else installed_lower
            has_newer = any(
                avail[0].lower().startswith(base_name) and avail[0].lower() != installed_lower
                for avail in available_models
            )
            if has_newer:
                obsolete.append(installed)

    return obsolete


def explain_recommendation(rec: ModelRecommendation) -> str:
    """Generate user-friendly explanation for a recommendation.

    Args:
        rec: Model recommendation

    Returns:
        Formatted explanation string
    """
    size_desc = {
        "1B": "Very small - fast, basic capabilities",
        "3B": "Small - good balance of speed and capability",
        "7B": "Medium - strong general performance",
        "8B": "Medium - strong general performance",
        "9B": "Medium - strong general performance",
        "13B": "Large - excellent performance, slower",
        "34B": "Very large - top performance, requires powerful hardware",
        "70B": "Massive - best quality, needs high-end hardware",
    }

    status = "✓ INSTALLED" if rec.is_installed else "Available to download"
    if rec.is_outdated:
        status = "⚠ OUTDATED - Update available"

    explanation = f"""
**{rec.model_name}** ({rec.size} parameters) - {status}
- {size_desc.get(rec.size, 'Model size: ' + rec.size)}
- {rec.reason}
- RAM needed: ~{rec.ram_required_gb:.1f}GB
- Performance: {rec.performance_tier.upper()}
"""

    return explanation.strip()
