# Coding-Focused LLM Models Leaderboard

Generated on: 2025-04-10

## Overview
This leaderboard shows the performance ranking of various Large Language Models specifically focused on coding tasks. Models are tested using the `coding101` promptset which includes challenging programming tasks, bug finding, and polyglot programming challenges.

## Overall Rankings

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1566 |
| 2 | cogito:32b | 1565 |
| 3 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1473 |
| 4 | phi4:latest | 1469 |
| 5 | qwen2.5-coder:32b | 1410 |
| 6 | deepseek-r1:32b | 1403 |
| 7 | deepcoder:latest | 1368 |
| 8 | llama3.3:70b-instruct-q2_K | 1366 |
| 9 | gemma3:4b | 1250 |
| 10 | llama3.1:8b | 1129 |

## Category Rankings

### Coding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1697 |
| 2 | cogito:32b | 1553 |
| 3 | phi4:latest | 1473 |
| 4 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1430 |
| 5 | gemma3:4b | 1380 |
| 6 | qwen2.5-coder:32b | 1338 |
| 7 | deepseek-r1:32b | 1334 |
| 8 | deepcoder:latest | 1326 |
| 9 | llama3.3:70b-instruct-q2_K | 1297 |
| 10 | llama3.1:8b | 1172 |

### BugFinding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | cogito:32b | 1599 |
| 2 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1494 |
| 3 | phi4:latest | 1484 |
| 4 | deepseek-r1:32b | 1452 |
| 5 | llama3.3:70b-instruct-q2_K | 1438 |
| 6 | gemma3:27b | 1432 |
| 7 | qwen2.5-coder:32b | 1424 |
| 8 | deepcoder:latest | 1391 |
| 9 | gemma3:4b | 1155 |
| 10 | llama3.1:8b | 1130 |

### Polyglot Programming

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1588 |
| 2 | cogito:32b | 1555 |
| 3 | mistral-small3.1:24b-instruct-2503-q4_K_M | 1499 |
| 4 | qwen2.5-coder:32b | 1457 |
| 5 | phi4:latest | 1454 |
| 6 | deepseek-r1:32b | 1424 |
| 7 | deepcoder:latest | 1390 |
| 8 | llama3.3:70b-instruct-q2_K | 1362 |
| 9 | gemma3:4b | 1192 |
| 10 | llama3.1:8b | 1079 |

## Analysis

### Key Insights from Latest Results

- **gemma3:27b** and **cogito:32b** are in an extremely tight race for the top spot (1566 vs 1565)
- **gemma3:27b** has further extended its lead in pure coding tasks, now at an exceptional 1697 ELO
- **cogito:32b**'s dominance in bug finding has grown stronger, reaching 1599 ELO
- The rankings show clear specialization patterns - different models excel in different coding domains
- **llama3.1:8b** has fallen further behind, now below 1130 ELO overall

### Model Specializations

| Model | Strongest Area | Rating Gap* | Analysis |
|-------|---------------|------------|----------|
| gemma3:27b | Coding | +131 | Extraordinary performance in pure coding tasks |
| cogito:32b | BugFinding | +44 | Security and debugging specialist |
| phi4:latest | Coding | +19 | Most balanced performance across categories |
| mistral-small | Polyglot | +69 | Strong cross-language capabilities |

*Rating gap: Difference between model's highest category rating and overall rating

### Tier Analysis

1. **Elite Tier (1550+)**: gemma3:27b, cogito:32b
   - These models demonstrate exceptional coding capabilities across the board
   - The gap to the next tier is substantial (90+ points)

2. **Strong Performers (1450-1550)**: mistral-small3.1, phi4:latest
   - Reliable models with balanced performance
   - Competitive in all categories

3. **Solid Tier (1350-1450)**: qwen2.5-coder, deepseek-r1, deepcoder, llama3.3
   - Good overall performance with specific strengths
   - Some specialized capabilities

4. **Entry Tier (<1350)**: gemma3:4b, llama3.1:8b
   - Significant gap from higher tiers
   - Struggle particularly in bug finding and polyglot coding

The data shows the importance of using specialized models for specific coding tasks. While gemma3:27b excels at writing code from scratch, cogito:32b is the clear choice for code review and bug identification.

*Generated with rank_llms using the coding101 promptset on 2025-04-10*