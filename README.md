# ALI - Assistente Locale Intelligente

Local AI assistant with screen reading and system control capabilities.

## Features

- 🤖 Local LLM integration via Ollama
- 👁️ Screen reading capabilities (planned)
- 🎮 System control and automation (planned)
- 💬 Text-based interaction
- 🔒 Fully local and private

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

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

5. Pull an Ollama model:
```bash
ollama pull llama2
```

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

## License

MIT
