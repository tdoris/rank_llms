# Rank LLMs

Rank-LLMs is a CLI tool for evaluating local LLMs via Ollama. It runs A/B tests on customizable prompt sets—letting you compare models on tasks that matter to you—and uses an AI judge to pick the better response. Results are scored with an Elo-based system and output as clear, side-by-side reports. Easy to extend with your own prompt sets for domain-specific evaluations.

Requires: ollama, Anthropic API key. 

## Features

- Compare models head-to-head with direct A/B evaluations
- Evaluate across different categories with challenging prompts in customizable promptsets
  - Default categories include:
    - General Knowledge
    - Programming
    - Reasoning
- Use Claude to judge which model's response is better
- Generate comprehensive comparison reports
- Build an ELO-based leaderboard for overall model ranking
- Generate Bradley-Terry model win probability matrices for any subset of models
- View side-by-side comparisons with winner explanations
- Automatically archive comparison results for reuse
- Get suggestions for additional tests to improve ranking confidence
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

### Compare Two Models

```bash
# Compare two models with 5 prompts per category (default)
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k

# Compare with fewer prompts for a quicker test
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --num-prompts 2

# Use a specific promptset
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --promptset coding101

# Specify a custom output file
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --output custom-results.md

# Force retesting of models even if archived results exist
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --force-retest

# Set a specific log level
rank-llms compare gemma3:27b llama3.1:70b-instruct-q2_k --log-level DEBUG
```

### View Promptset Information

```bash
# Display information about the default promptset
rank-llms promptset-info

# Display information about a specific promptset
rank-llms promptset-info --promptset coding101
```

### Suggest Additional Tests

```bash
# Suggest additional tests to improve ranking confidence
rank-llms suggest-tests

# Suggest tests for a specific promptset
rank-llms suggest-tests --promptset coding101

# Limit the number of suggestions
rank-llms suggest-tests --max-suggestions 5
```

### Generate Leaderboard

```bash
# Generate and display the leaderboard
rank-llms leaderboard

# Specify a promptset for the leaderboard
rank-llms leaderboard --promptset coding101

# Force refreshing of all ELO ratings from comparison results
rank-llms leaderboard --force-refresh

# Specify custom output paths
rank-llms leaderboard --output custom-leaderboard.md --json-output custom-leaderboard.json
```

### Rank a Subset of Models (Bradley-Terry Model)

```bash
# Rank a subset of models and display their win probabilities using Bradley-Terry model
rank-llms ranksubset gemma3:27b cogito:32b gemma3:12b phi4:latest

# Specify a promptset for the analysis
rank-llms ranksubset gemma3:27b cogito:32b --promptset coding101

# Save the results to a file
rank-llms ranksubset gemma3:27b cogito:32b phi4:latest --output bt-ranking.md
```

## Requirements

- Python 3.8+
- Ollama (with models installed)
- Anthropic API key (set as ANTHROPIC_API_KEY environment variable)

## Output

- Detailed comparison reports with side-by-side model responses
- Win percentage summaries for each category
- ELO-based leaderboard in markdown and JSON formats
- Log files stored in the `logs` directory with timestamped filenames

## Promptsets and Archiving System

The app uses customizable promptsets stored in JSON files:

- Promptsets are located in the `promptsets` directory
- Each promptset is a JSON file containing categories and prompts
- The default promptset is `basic1.json`
- A programming-specific promptset called `coding101.json` is included
- You can create your own promptsets by adding new JSON files
- Use the `--promptset` option to specify which promptset to use
- Use the `promptset-info` command to view the contents of a promptset

The app maintains an archive of model comparisons in the `test_archive/<promptset>/comparisons` directory:

- Comparisons are organized by promptset
- Each comparison is stored in a separate JSON file named after the two models
- When you request a comparison that has already been performed, the results are loaded from the archive
- This allows for building a comprehensive leaderboard without repeating tests
- The `--force-retest` flag can be used to force a new comparison even if archived results exist

## ELO Rating System

The app uses an ELO rating system to rank models based on head-to-head comparisons:

- Each model starts with a default rating (1400)
- Ratings are updated after each comparison based on win/loss results
- The leaderboard displays models ranked by their ELO rating
- Separate ELO ratings are maintained for each category
- Separate leaderboards are maintained for each promptset
- The system automatically builds leaderboards from all comparison results

## Logging System

The app includes a comprehensive logging system that records all operations:

- Each run creates a timestamped log file in the `logs` directory
- Logs include information about:
  - Models being compared
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
