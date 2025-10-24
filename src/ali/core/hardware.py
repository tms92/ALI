"""Hardware detection for model recommendations."""

import platform
import subprocess
from dataclasses import dataclass

from loguru import logger


@dataclass
class HardwareInfo:
    """System hardware information."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    platform: str


def get_ram_info() -> tuple[float, float]:
    """Get total and available RAM in GB.

    Returns:
        Tuple of (total_ram_gb, available_ram_gb)
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        return total_gb, available_gb
    except ImportError:
        logger.warning("psutil not available, using fallback RAM detection")
        return 8.0, 4.0  # Conservative fallback


def get_gpu_info() -> tuple[bool, str | None, float | None]:
    """Detect GPU availability and specifications.

    Returns:
        Tuple of (gpu_available, gpu_name, vram_gb)
    """
    # Try NVIDIA GPU first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                gpu_name = parts[0].strip()
                vram_str = parts[1].strip().split()[0]  # "11441 MiB" -> "11441"
                vram_gb = float(vram_str) / 1024
                logger.info(f"Detected NVIDIA GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return True, gpu_name, vram_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"NVIDIA GPU detection failed: {e}")

    # Try AMD GPU (ROCm)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            logger.info("Detected AMD GPU")
            return True, "AMD GPU", None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Try Metal (macOS)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if "Metal" in result.stdout or "Apple" in result.stdout:
                logger.info("Detected Metal GPU (Apple Silicon)")
                return True, "Apple GPU", None
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    logger.info("No GPU detected, CPU-only mode")
    return False, None, None


def detect_hardware() -> HardwareInfo:
    """Detect system hardware specifications.

    Returns:
        HardwareInfo object with system specifications
    """
    logger.info("Detecting system hardware...")

    total_ram, available_ram = get_ram_info()
    cpu_count = platform.os.cpu_count() or 1
    gpu_available, gpu_name, gpu_vram = get_gpu_info()

    hardware = HardwareInfo(
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        cpu_count=cpu_count,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        platform=platform.system(),
    )

    logger.info(f"Hardware: {total_ram:.1f}GB RAM, {cpu_count} CPUs, " f"GPU: {gpu_name or 'None'}")

    return hardware
