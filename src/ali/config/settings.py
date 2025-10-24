"""Application settings and configuration management."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""

    host: str = Field(default="http://localhost:11434", description="Ollama API host")
    model: str = Field(default="llama2", description="Default model to use")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    temperature: float = Field(default=0.7, description="Model temperature", ge=0.0, le=2.0)
    context_window: int = Field(default=4096, description="Context window size")


class VisionSettings(BaseSettings):
    """Screen capture and vision configuration."""

    screenshot_quality: int = Field(default=85, description="JPEG quality (1-100)", ge=1, le=100)
    max_image_size: tuple[int, int] = Field(
        default=(1920, 1080), description="Maximum image dimensions"
    )
    vision_model: str = Field(default="llava", description="Vision-capable model")


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="ALI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Paths
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "ali",
        description="Configuration directory",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "ali",
        description="Data directory",
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ali",
        description="Cache directory",
    )

    # Modules
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.config_dir, self.data_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        _settings.ensure_directories()
    return _settings
