# 14B-Scale Model Comparison: Bradley-Terry Analysis

This analysis shows head-to-head win probabilities for similar-sized (~12-14B parameter) models on the coding101 promptset. The Bradley-Terry model estimates the probability of each model beating every other model based on actual test comparisons.

## Win Probability Matrix

Estimated probability of row model beating column model:

| Model | cogito:14b | deepseek-r1:14b | gemma3:12b | phi4:latest |
|-------|-------|-------|-------|-------|
| **cogito:14b** | - | 0.720 | 0.794 | 0.929 |
| **deepseek-r1:14b** | 0.280 | - | 0.599 | 0.836 |
| **gemma3:12b** | 0.206 | 0.401 | - | 0.773 |
| **phi4:latest** | 0.071 | 0.164 | 0.227 | - |

## Model Strength Parameters

| Rank | Model | Strength |
|------|-------|----------|
| 1 | cogito:14b | 58.00 |
| 2 | deepseek-r1:14b | 22.53 |
| 3 | gemma3:12b | 15.06 |
| 4 | phi4:latest | 4.42 |

## Key Insights

- **cogito:14b** clearly dominates in this comparison group:
  - Has >70% chance of winning against all competitors
  - Nearly 93% chance of beating phi4:latest
  - Almost 3x the strength rating of the second-place model

- **deepseek-r1:14b**, despite ranking lower in the ELO leaderboard, shows strong performance in direct comparisons:
  - ~60% chance of beating gemma3:12b
  - ~84% chance of beating phi4:latest

- **gemma3:12b** outperforms phi4:latest with a 77% win probability, but struggles against the top two models

- **phi4:latest** has the lowest strength parameter and is predicted to lose to all other models in this subset

*Generated on 2025-04-11 using the rank-llms Bradley-Terry analysis on the coding101 promptset*
