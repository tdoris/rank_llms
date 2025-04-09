# LLM Models Leaderboard

Generated on: 2025-04-09

## Overview
This leaderboard shows the performance ranking of various Large Language Models tested using the `rank_llms` tool. Models are compared head-to-head with Claude acting as a judge, and an ELO rating system is used to rank them.

## Overall Rankings

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1422 |
| 2 | gemma3:4b | 1414 |
| 3 | qwen2.5:32b | 1397 |
| 4 | phi4:latest | 1393 |
| 5 | llama3.1:8b | 1374 |

## Category Rankings

### General Knowledge

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1432 |
| 2 | gemma3:4b | 1430 |
| 3 | qwen2.5:32b | 1384 |
| 4 | phi4:latest | 1384 |
| 5 | llama3.1:8b | 1370 |

### Programming

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1425 |
| 2 | gemma3:4b | 1413 |
| 3 | qwen2.5:32b | 1397 |
| 4 | phi4:latest | 1393 |
| 5 | llama3.1:8b | 1372 |

### Reasoning

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1422 |
| 2 | qwen2.5:32b | 1410 |
| 3 | phi4:latest | 1402 |
| 4 | gemma3:4b | 1392 |
| 5 | llama3.1:8b | 1375 |

### Creative Writing

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1413 |
| 2 | gemma3:4b | 1413 |
| 3 | llama3.1:8b | 1375 |

### Summarization

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1419 |
| 2 | gemma3:4b | 1406 |
| 3 | llama3.1:8b | 1375 |

## Analysis

- **gemma3:27b** consistently leads across all categories, showcasing strong overall performance
- **gemma3:4b** performs remarkably well for its size, often coming close to its larger sibling
- **qwen2.5:32b** and **phi4:latest** are in the middle tier, with phi4 showing particularly strong reasoning capabilities 
- **llama3.1:8b** currently ranks lowest in all categories but still maintains competitive ratings

The models were compared using the `basic1` promptset. Future comparisons using the new `coding101` promptset will provide more insight into programming-specific capabilities.

*Generated with rank_llms*