# Coding-Focused LLM Models Leaderboard

Generated on: 2025-04-10

## Overview
This leaderboard shows the performance ranking of various Large Language Models specifically focused on coding tasks. Models are tested using the `coding101` promptset which includes challenging programming tasks, bug finding, and polyglot programming challenges.

## Overall Rankings

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1554 |
| 2 | cogito:32b | 1552 |
| 3 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1475 |
| 4 | phi4:latest | 1470 |
| 5 | qwen2.5-coder:32b | 1408 |
| 6 | deepseek-r1:32b | 1401 |
| 7 | deepcoder:latest | 1376 |
| 8 | llama3.3:70b-instruct-q2_K | 1365 |
| 9 | gemma3:4b | 1254 |
| 10 | llama3.1:8b | 1145 |

## Category Rankings

### Coding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1675 |
| 2 | cogito:32b | 1522 |
| 3 | phi4:latest | 1485 |
| 4 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1438 |
| 5 | gemma3:4b | 1375 |
| 6 | deepcoder:latest | 1343 |
| 7 | qwen2.5-coder:32b | 1339 |
| 8 | deepseek-r1:32b | 1329 |
| 9 | llama3.3:70b-instruct-q2_K | 1307 |
| 10 | llama3.1:8b | 1187 |

### BugFinding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | cogito:32b | 1582 |
| 2 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1499 |
| 3 | phi4:latest | 1491 |
| 4 | deepseek-r1:32b | 1449 |
| 5 | gemma3:27b | 1426 |
| 6 | llama3.3:70b-instruct-q2_K | 1425 |
| 7 | qwen2.5-coder:32b | 1424 |
| 8 | deepcoder:latest | 1393 |
| 9 | gemma3:4b | 1169 |
| 10 | llama3.1:8b | 1142 |

### Polyglot Programming

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1575 |
| 2 | cogito:32b | 1564 |
| 3 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1491 |
| 4 | qwen2.5-coder:32b | 1452 |
| 5 | phi4:latest | 1436 |
| 6 | deepseek-r1:32b | 1425 |
| 7 | deepcoder:latest | 1395 |
| 8 | llama3.3:70b-instruct-q2_K | 1364 |
| 9 | gemma3:4b | 1198 |
| 10 | llama3.1:8b | 1099 |

## Analysis

- **gemma3:27b** dominates overall, with an exceptional performance in the Coding category (1675 ELO)
- **cogito:32b** is a very strong contender, particularly excelling in BugFinding where it ranks #1
- **mistral-small3.1** and **phi4:latest** form a solid second tier, with both showing balanced performance across categories
- **Specialized models** like qwen2.5-coder:32b and deepseek-r1:32b perform well but can't match the top performers
- There's a significant gap between the top 8 models and the smaller **gemma3:4b** and **llama3.1:8b** models (100+ points)
- **BugFinding** shows the most variation in model ranking, suggesting this skill varies significantly between models

Notably, different models excel in different categories:
- **gemma3:27b** dominates pure coding tasks
- **cogito:32b** leads in finding bugs and vulnerabilities
- The top models are all competitive in polyglot programming, with less variation in this category

The data shows the importance of testing across multiple coding tasks, as performance varies significantly by category. The comprehensive testing done using the `coding101` promptset provides a more nuanced understanding of each model's strengths and weaknesses in programming domains.

*Generated with rank_llms using the coding101 promptset*