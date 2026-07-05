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
  - Counters judge **position bias** by scoring each comparison in both A/B orderings (only agreement counts as a win); disable with `--no-counter-position-bias`
  - Configurable judge model (`--judge-model`, defaults to `claude-opus-4-8`) and self-consistency voting (`--judge-samples`)
  - Estimate judge API calls and cost before a run with `--dry-run`
- Generate comprehensive comparison reports
- Build an ELO-based leaderboard for overall model ranking
- Compare any subset of models using direct head-to-head results
- Create focus-based rankings centered on a specific model
- Rank models within any single category (`categoryrank <category> ...`; `codingrank` is the Coding alias), with **Wilson 95% confidence intervals** and flags for statistically indistinguishable models
- Export rankings to self-contained HTML or JSON (`--export-html` / `--export-json`)
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

### Rank a Subset of Models (Direct Comparison)

```bash
# Rank a subset of models based on actual head-to-head comparison results
rank-llms ranksubset gemma3:27b cogito:32b gemma3:12b phi4:latest

# Specify a promptset for the analysis
rank-llms ranksubset gemma3:27b cogito:32b --promptset coding101

# Save the results to a file
rank-llms ranksubset gemma3:27b cogito:32b phi4:latest --output model-comparison.md
```

If any of the models haven't been directly compared, the command will show you what comparisons are missing and provide the commands to run them.

### Focus-Based Model Ranking

```bash
# Rank models based on their performance against a specific focus model
rank-llms focusrank gemma3:27b

# Use a specific promptset
rank-llms focusrank gemma3:27b --promptset coding101

# Limit to direct comparisons only (no transitive relationships)
rank-llms focusrank gemma3:27b --depth 1

# Save the results to a file
rank-llms focusrank gemma3:27b --output gemma27b-focus-ranking.md
```

Focus ranking uses win ratios (other_model_wins / focus_model_wins) to rank models relative to the focus model. A ratio > 1.0 means the model outperforms the focus model.

### Coding-Specific Analysis

```bash
# Generate coding-specific rankings (implementation tasks only)
rank-llms codingrank phi4:latest gemma3:12b qwen2.5-coder:14b cogito:14b deepseek-r1:14b

# Specify a different output file
rank-llms codingrank phi4:latest gemma3:12b qwen2.5-coder:14b --output coding_analysis.md

# Use a different test archive directory
rank-llms codingrank phi4:latest gemma3:12b --archive /path/to/test_archive
```

The codingrank command analyzes only the Coding category tasks from the coding101 promptset, ignoring other categories like BugFinding and Polyglot. This focuses on pure code implementation capability versus overall programming skills.

## Example: Ranking New Models on the Leaderboard (basic1)

Two strong models were added to the `basic1` set and evaluated head-to-head against the field. Every comparison was judged by `claude-opus-4-8` with position-bias counter-balancing (each pair scored in both A/B orderings; only agreement counts as a win), at `--num-prompts 3` (9 prompts per matchup across General Knowledge, Programming, and Reasoning).

### qwen3.6:27b — how ELO catches up to direct results

When `qwen3.6:27b` was first added (four opponents), it went **34–1–1**, yet landed only **#2 by ELO (1458)** behind `gemma3:27b` (1470) — because ELO is path-dependent and `gemma3:27b` had accumulated its rating over many more games. The direct win rate (≈94%) said it was already the strongest model, and the note here predicted "the ELO gap should close as it plays more models."

That is exactly what happened. After `gemma4:26b` joined the field (below), `qwen3.6:27b`'s record stands at **39–3–3** across 45 prompts and it now sits at a commanding **#1 overall (ELO 1622)** — #1 in every tested category. This is the discrepancy the `focusrank` and `ranksubset` direct-comparison commands exist to surface: direct win rate leads, ELO follows as the comparison graph fills in.

### gemma4:26b — dominates the field, loses to the champion

`gemma4:26b` was then added against five opponents (the four above plus a direct `qwen3.6:27b` head-to-head). Overall record: **35–7–3** (≈78%).

| Opponent | gemma4 W–L–T |
|----------|--------------|
| gemma3:latest | 9–0–0 |
| phi4:latest | 8–0–1 |
| gemma3:27b | 8–1–0 |
| cogito:14b | 8–1–0 |
| **qwen3.6:27b** | **2–5–2** |

By category: **Reasoning 12–0–3** (never lost a reasoning prompt), General Knowledge 12–3–0, Programming 11–4–0.

`gemma4:26b` beats every model in the set *except* `qwen3.6:27b`, which won their series decisively — cleanly separating the top two. It lands **#2 overall (ELO 1520)**: #2 in General Knowledge and Reasoning, #3 in Programming.

**Current basic1 top of the leaderboard:**

| Rank | Model | ELO |
|------|-------|-----|
| 1 | qwen3.6:27b | 1622 |
| 2 | gemma4:26b | 1520 |
| 3 | gemma3:27b | 1453 |
| 4 | qwen2.5:32b | 1450 |
| 5 | phi4:latest | 1352 |

Note that adding a strong new model reshuffles existing ratings: `gemma3:27b` fell from #1 to #3 once `qwen3.6:27b` and `gemma4:26b` entered, because ELO is recomputed from the full game history on every leaderboard rebuild.

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
