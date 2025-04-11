# 14B-Scale Model Comparison: Bradley-Terry Analysis

This analysis shows head-to-head win probabilities for similar-sized (~12-14B parameter) models on the coding101 promptset. The Bradley-Terry model estimates the probability of each model beating every other model based on actual test comparisons.

## Win Probability Matrix (Bradley-Terry Model)

Estimated probability of row model beating column model:

| Model | cogito:14b | deepseek-r1:14b | gemma3:12b | phi4:latest |
|-------|-------|-------|-------|-------|
| **cogito:14b** | - | 0.720 | 0.794 | 0.929 |
| **deepseek-r1:14b** | 0.280 | - | 0.599 | 0.836 |
| **gemma3:12b** | 0.206 | 0.401 | - | 0.773 |
| **phi4:latest** | 0.071 | 0.164 | 0.227 | - |

## Actual Head-to-Head Results

### cogito:14b vs gemma3:12b
- **Overall**: cogito:14b wins 53.3% (8/15), gemma3:12b wins 40.0% (6/15), Ties 6.7% (1/15)
- **Coding**: cogito:14b wins 40.0% (2/5), gemma3:12b wins 60.0% (3/5), Ties 0.0% (0/5)
- **BugFinding**: cogito:14b wins 80.0% (4/5), gemma3:12b wins 20.0% (1/5), Ties 0.0% (0/5)
- **Polyglot**: cogito:14b wins 40.0% (2/5), gemma3:12b wins 40.0% (2/5), Ties 20.0% (1/5)

### Model Strength Parameters (Bradley-Terry)

| Rank | Model | Strength |
|------|-------|----------|
| 1 | cogito:14b | 58.00 |
| 2 | deepseek-r1:14b | 22.53 |
| 3 | gemma3:12b | 15.06 |
| 4 | phi4:latest | 4.42 |

## Key Insights (Revised)

- **cogito:14b vs gemma3:12b** is much closer than the Bradley-Terry model suggests:
  - Actual win rate is 53.3% vs the model's estimated 79.4%
  - gemma3:12b actually outperforms cogito:14b in Coding tasks (60% win rate)
  - cogito:14b dominates in BugFinding (80% win rate)
  - They are evenly matched in Polyglot tasks

- **cogito:14b** shows strong performance overall:
  - Notable strength in debugging tasks
  - Not as dominant in implementation tasks as the model suggests

- **gemma3:12b** is stronger than the Bradley-Terry model indicates:
  - Particularly excels in Coding implementation
  - More competitive with cogito:14b than strength parameters suggest
  
- **Note on Model Discrepancy**: The Bradley-Terry model overestimates cogito:14b's advantage over gemma3:12b, possibly due to how it weights transitive relationships between all models or due to category-specific strengths being smoothed out in the overall model.

- **phi4:latest** still appears to be the weakest in this subset

*Generated on 2025-04-11 using the rank-llms Bradley-Terry analysis on the coding101 promptset*
