# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - TBD

### Added
- Smart onboarding flow with interactive model selection (#1)
- Ollama auto-installer with cross-platform support
- Initial project structure
- Python package configuration with pyproject.toml
- Pydantic-based settings system
- Ollama client wrapper with chat and streaming support
- Ollama service lifecycle management
- Hardware detection (CPU, RAM, GPU - NVIDIA/AMD/Metal)
- Model recommendation system based on hardware capabilities
- Ollama installer with cross-platform support (Linux auto-install, Windows/macOS manual)
- Comprehensive unit test suite (73 tests, 65% coverage)
- GitHub Actions CI/CD pipeline
- Test pyramid strategy with unit/integration/E2E planning
- Professional testing framework with markers and coverage tracking

### Testing
- Unit tests: 73 tests, 65% coverage
  - Config module: 100% coverage (10 tests)
  - Recommendations module: 100% coverage (16 tests)
  - Installer module: 79% coverage (25 tests)
  - Services module: 64% coverage (14 tests)
  - Client module: 64% coverage (9 tests)
- Integration tests: Planned for v0.2.0 (tracked in #7)
- E2E tests: Planned for v1.0.0

### Documentation
- README with setup instructions and roadmap
- Testing strategy documented in directives
- Versioning strategy established
- GitHub project organization (milestones, labels)

[Unreleased]: https://github.com/tms92/ALI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tms92/ALI/releases/tag/v0.1.0
