# ALI - Assistente Locale Intelligente

[![Tests](https://github.com/tms92/ALI/actions/workflows/test.yml/badge.svg)](https://github.com/tms92/ALI/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Local AI assistant with smart onboarding and Ollama integration (in active development).

## Why ALI?

ALI makes local Large Language Models accessible to everyone, regardless of technical expertise. By running AI models entirely on your own machine, you can:

- **Protect your privacy** - Work with sensitive personal data without sending it to external servers
- **Maintain control** - Your conversations, documents, and information never leave your device
- **Break down barriers** - No technical knowledge required - ALI handles setup and configuration automatically
- **Use AI confidently** - No subscription fees, no data collection, no privacy concerns

Whether you're handling medical records, financial documents, or personal communications, ALI ensures your data stays private while giving you access to powerful AI capabilities.

## Current Features

- 🚀 **Smart onboarding** - First-run setup that guides you through installation
- 🤖 **Ollama auto-installer** - Detects and installs Ollama automatically (Linux)
- 💻 **Hardware detection** - Detects CPU, RAM, and GPU capabilities
- ⚙️ **Smart model selection** - Recommends appropriate models for your hardware
- 🔧 **Auto-start Ollama** - Detects and starts Ollama service automatically
- 🔒 **Fully local and private** - Your data never leaves your machine
- 🧪 **Comprehensive testing** - 91 tests with 58% coverage

## In Development

- 💬 **Interactive chat** - Conversational interface ([#9](https://github.com/tms92/ALI/issues/9))
- 👁️ **Screen reading** - Capture and analyze screen content ([#3](https://github.com/tms92/ALI/issues/3))
- 🎮 **System control** - Keyboard/mouse automation ([#4](https://github.com/tms92/ALI/issues/4))
- 🖥️ **Desktop GUI** - System tray and overlay interface ([#5](https://github.com/tms92/ALI/issues/5))

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) - ALI will help install/start it automatically on first run

## Setup

1. Clone the repository:
```bash
git clone https://github.com/tms92/ALI.git
cd ALI
```

2. Create virtual environment:
```bash
python -m venv .venv
```

3. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -e ".[dev]"
```

5. Run ALI - it will guide you through Ollama setup and model selection:
```bash
ali
```

On first run, ALI will:
- Check if Ollama is installed
- Start Ollama service if needed
- Help you choose an appropriate model for your hardware

## Usage

Run ALI:
```bash
ali
```

Or directly:
```bash
python -m ali.main
```

## Project Structure

```
ali/
├── src/ali/
│   ├── config/           # Configuration management
│   ├── core/             # Core functionality
│   │   ├── hardware.py   # Hardware detection (CPU, RAM, GPU)
│   │   ├── installer.py  # Ollama installer utilities
│   │   ├── onboarding.py # Interactive onboarding flow
│   │   ├── recommendations.py # Model recommendation system
│   │   └── services.py   # Ollama service management
│   ├── llm/              # LLM integration
│   │   └── client.py     # Ollama client wrapper
│   ├── vision/           # Screen reading (planned)
│   └── control/          # System control (planned)
├── tests/
│   └── unit/             # Unit tests (91 tests, 58% coverage)
└── pyproject.toml        # Project configuration
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
ruff check src/
```

Type checking:
```bash
mypy src/
```

## Contributing

See open [issues](https://github.com/tms92/ALI/issues) for planned features and improvements.

Current roadmap:
- [x] Foundation and Ollama integration
- [x] CI/CD pipeline ([#2](https://github.com/tms92/ALI/issues/2))
- [x] Smart onboarding ([#1](https://github.com/tms92/ALI/issues/1))
- [ ] Interactive chat ([#9](https://github.com/tms92/ALI/issues/9))
- [ ] Screen reading ([#3](https://github.com/tms92/ALI/issues/3))
- [ ] System control ([#4](https://github.com/tms92/ALI/issues/4))
- [ ] Desktop GUI ([#5](https://github.com/tms92/ALI/issues/5))

## License

MIT
