# Coding-Focused LLM Models Leaderboard

Generated on: 2025-04-09

## Overview
This leaderboard shows the performance ranking of various Large Language Models specifically focused on coding tasks. Models are tested using the `coding101` promptset which includes challenging programming tasks, bug finding, and polyglot programming challenges.

## Overall Rankings

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1479 |
| 2 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1453 |
| 3 | phi4:latest | 1453 |
| 4 | cogito:32b | 1433 |
| 5 | qwen2.5-coder:32b | 1415 |
| 6 | deepseek-r1:32b | 1414 |
| 7 | gemma3:4b | 1305 |
| 8 | llama3.1:8b | 1248 |

## Category Rankings

### Coding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1536 |
| 2 | phi4:latest | 1441 |
| 3 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1430 |
| 4 | cogito:32b | 1402 |
| 5 | gemma3:4b | 1395 |
| 6 | qwen2.5-coder:32b | 1370 |
| 7 | deepseek-r1:32b | 1340 |
| 8 | llama3.1:8b | 1286 |

### BugFinding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | phi4:latest | 1493 |
| 2 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1472 |
| 3 | cogito:32b | 1471 |
| 4 | deepseek-r1:32b | 1449 |
| 5 | qwen2.5-coder:32b | 1431 |
| 6 | gemma3:27b | 1397 |
| 7 | gemma3:4b | 1259 |
| 8 | llama3.1:8b | 1228 |

### Polyglot Programming

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1505 |
| 2 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1459 |
| 3 | deepseek-r1:32b | 1452 |
| 4 | qwen2.5-coder:32b | 1444 |
| 5 | cogito:32b | 1428 |
| 6 | phi4:latest | 1427 |
| 7 | gemma3:4b | 1257 |
| 8 | llama3.1:8b | 1228 |

## Analysis

- **gemma3:27b** leads overall, with exceptional performance in general coding tasks and polyglot programming
- **phi4:latest** excels in bug finding tasks, showing strong capabilities in code review and debugging
- **mistral-small3.1:24b-instruct** performs consistently well across all coding categories
- **Specialized models** like qwen2.5-coder:32b, cogito:32b, and deepseek-r1:32b show strong performance, especially in bug finding
- The smaller **gemma3:4b** and **llama3.1:8b** models fall behind in these challenging coding tasks

The models were compared using the `coding101` promptset, which includes more challenging and specific programming-related tasks than the general `basic1` promptset. This provides greater insight into the models' abilities to handle complex coding challenges, debug code, and work with multiple programming languages.

*Generated with rank_llms using the coding101 promptset*