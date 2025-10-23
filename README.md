# ALI - Assistente Locale Intelligente

[![Tests](https://github.com/tms92/ALI/actions/workflows/test.yml/badge.svg)](https://github.com/tms92/ALI/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Local AI assistant with screen reading and system control capabilities.

## Features

- 🤖 **Local LLM integration** via Ollama with automatic service management
- 🔧 **Auto-start Ollama** - ALI detects and starts Ollama automatically
- ⚙️ **Smart model selection** - Recommends models based on your hardware
- 💬 **Text-based interaction** - Clean conversational interface
- 🔒 **Fully local and private** - Your data never leaves your machine
- 🧪 **Comprehensive testing** - 32 tests with 55% coverage

### Planned Features
- 👁️ Screen reading capabilities ([#3](https://github.com/tms92/ALI/issues/3))
- 🎮 System control and automation ([#4](https://github.com/tms92/ALI/issues/4))
- 🖥️ Desktop GUI with system tray ([#5](https://github.com/tms92/ALI/issues/5))

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
│   ├── config/       # Configuration management
│   ├── core/         # Core functionality
│   ├── llm/          # LLM integration
│   ├── vision/       # Screen reading (planned)
│   └── control/      # System control (planned)
├── tests/            # Test suite
└── pyproject.toml    # Project configuration
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
- [x] CI/CD pipeline
- [ ] Smart onboarding ([#1](https://github.com/tms92/ALI/issues/1))
- [ ] Screen reading ([#3](https://github.com/tms92/ALI/issues/3))
- [ ] System control ([#4](https://github.com/tms92/ALI/issues/4))
- [ ] Desktop GUI ([#5](https://github.com/tms92/ALI/issues/5))

## License

MIT
