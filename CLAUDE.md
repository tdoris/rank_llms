# CLAUDE.md - Guide for Agentic Coding Assistants

## Build & Test Commands
- Install: `pip install -e .`
- Compare models: `rank-llms compare <model1> <model2> --num-prompts 5`
- Example: `rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --num-prompts 3`
- Force retest: `rank-llms compare <model1> <model2> --force-retest`
- Generate leaderboard: `rank-llms leaderboard`
- Set log level: `rank-llms compare <model1> <model2> --log-level DEBUG`
- Run tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_file.py::test_function`
- Lint code: `flake8 rank_llms/` or `ruff check rank_llms/`
- Type check: `mypy rank_llms/`

## Code Style Guidelines
- **Formatting**: Use Black with 88 character line limit
- **Imports**: Group by standard library, third-party, and local; sort alphabetically
- **Types**: Use type hints for all function parameters and return values
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Documentation**: Docstrings for all public functions (Google style)
- **Error Handling**: Use specific exception types with context-rich messages
- **Models**: Use Pydantic for data models and validation
- **CLI**: Use Typer for command-line interfaces
- **Console Output**: Use Rich for formatted terminal output