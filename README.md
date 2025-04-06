# Rank LLMs

A Python tool to test and rank the quality of responses from local LLMs running via Ollama.

## Features

- Test multiple Ollama models with one command
- Evaluate responses across 5 different categories:
  - General Knowledge
  - Creative Writing  
  - Programming
  - Reasoning
  - Summarization
- Score responses using Claude's evaluation
- Generate comprehensive markdown report
- View summary table in the console
- Automatically archive test results for faster comparisons
  - Full test results (with all 10 prompts per category) are saved
  - Archived results are automatically reused in future tests
  - Includes timestamp for when each model was evaluated
- Comprehensive logging system
  - Logs all operations to timestamped log files
  - Tracks Anthropic API calls and errors
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Installation

1. Ensure you have Ollama installed and models you want to test are downloaded
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rank_llms.git
   cd rank_llms
   ```

3. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Linux/macOS
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

6. Set up your Anthropic API key:
   ```bash
   # Linux/macOS
   export ANTHROPIC_API_KEY="your-api-key-here"
   
   # Windows (Command Prompt)
   set ANTHROPIC_API_KEY=your-api-key-here
   
   # Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   ```

## Usage

```bash
# Test multiple models with 10 prompts per category
rank-llms gemma3:27b llama3.1:70b-instruct-q2_k

# Test with fewer prompts for a quicker test
rank-llms gemma3:27b llama3.1:70b-instruct-q2_k --num-prompts 2

# Specify a custom output file
rank-llms gemma3:27b llama3.1:70b-instruct-q2_k --output custom-results.md

# Force retesting of models even if archived results exist
rank-llms gemma3:27b llama3.1:70b-instruct-q2_k --force-retest

# Set a specific log level
rank-llms gemma3:27b --log-level DEBUG
```

Alternatively, you can run the module directly:
```bash
# Make sure your virtual environment is activated
python -m rank_llms gemma3:27b llama3.1:70b-instruct-q2_k --num-prompts 3
```

## Requirements

- Python 3.8+
- Ollama (with models installed)
- Anthropic API key (set as ANTHROPIC_API_KEY environment variable)

## Output

- Detailed markdown results file with all prompts, responses, and evaluations
- Summary table in console showing average scores by category and response times
- Archived test results stored in the `test_archive` directory
- Log files stored in the `logs` directory with timestamped filenames

## Test Archive

The app maintains an archive of full test results (when using all 10 prompts per category) in the `test_archive` directory:

- Each model's results are stored in a separate file named after the model
- When you request a test with models that have already been tested with the full 10 prompts per category, their results are loaded from the archive
- This allows for quick comparisons of models without retesting
- The `--force-retest` flag can be used to force the app to retest models even if they have archived results
- Tests with fewer than 10 prompts per category will always execute the tests again (archives are only used for full tests)
- Each archived response includes a timestamp indicating when it was tested

## Logging System

The app includes a comprehensive logging system that records all operations:

- Each run creates a timestamped log file in the `logs` directory
- Logs include information about:
  - Models being tested
  - Prompts being processed
  - API calls to Ollama and Anthropic
  - Errors and warnings during execution
  - Archive operations
- Log levels can be configured with the `--log-level` parameter:
  - `DEBUG`: Detailed debugging information
  - `INFO`: General operational information (default)
  - `WARNING`: Issues that might cause problems
  - `ERROR`: Error conditions preventing functionality
  - `CRITICAL`: Critical errors that might cause the program to crash
- All logs also appear in the console output