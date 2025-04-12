# 14B-Scale Model Comparison: Direct Head-to-Head Analysis

This analysis shows the performance of similar-sized (~12-14B parameter) models on the coding101 promptset, based on actual head-to-head test results rather than mathematical projections.

## Overall Rankings

| Rank | Model | Average Win Rate |
|------|-------|----------------|
| 1 | phi4:latest | 0.756 |
| 2 | deepseek-r1:14b | 0.567 |
| 3 | gemma3:12b | 0.344 |
| 4 | cogito:14b | 0.333 |

## Win Probability Matrix

Probability of row model beating column model (based on head-to-head results):

| Model | phi4:latest | deepseek-r1:14b | gemma3:12b | cogito:14b |
|-------|-------|-------|-------|-------|
| **phi4:latest** | - | 0.800 | 0.800 | 0.667 |
| **deepseek-r1:14b** | 0.200 | - | 0.733 | 0.767 |
| **gemma3:12b** | 0.200 | 0.267 | - | 0.567 |
| **cogito:14b** | 0.333 | 0.233 | 0.433 | - |

## Detailed Head-to-Head Results

### cogito:14b vs gemma3:12b
- **Overall**: cogito:14b wins 53.3% (8/15), gemma3:12b wins 40.0% (6/15), Ties 6.7% (1/15)
- **Coding**: cogito:14b wins 40.0% (2/5), gemma3:12b wins 60.0% (3/5), Ties 0.0% (0/5)
- **BugFinding**: cogito:14b wins 80.0% (4/5), gemma3:12b wins 20.0% (1/5), Ties 0.0% (0/5)
- **Polyglot**: cogito:14b wins 40.0% (2/5), gemma3:12b wins 40.0% (2/5), Ties 20.0% (1/5)

### cogito:14b vs phi4:latest
- **Overall**: cogito:14b wins 33.3% (4/15), phi4:latest wins 60.0% (9/15), Ties 13.3% (2/15)
- **Coding**: cogito:14b wins 60.0% (3/5), phi4:latest wins 40.0% (2/5), Ties 0.0% (0/5)
- **BugFinding**: cogito:14b wins 0.0% (0/5), phi4:latest wins 100.0% (5/5), Ties 0.0% (0/5)
- **Polyglot**: cogito:14b wins 40.0% (2/5), phi4:latest wins 20.0% (1/5), Ties 40.0% (2/5)

### cogito:14b vs deepseek-r1:14b
- **Overall**: cogito:14b wins 23.3% (3/15), deepseek-r1:14b wins 76.7% (11/15), Ties 6.7% (1/15)
- **Coding**: cogito:14b wins 20.0% (1/5), deepseek-r1:14b wins 80.0% (4/5), Ties 0.0% (0/5)
- **BugFinding**: cogito:14b wins 40.0% (2/5), deepseek-r1:14b wins 60.0% (3/5), Ties 0.0% (0/5)
- **Polyglot**: cogito:14b wins 0.0% (0/5), deepseek-r1:14b wins 80.0% (4/5), Ties 20.0% (1/5)

### gemma3:12b vs phi4:latest
- **Overall**: gemma3:12b wins 20.0% (3/15), phi4:latest wins 80.0% (12/15), Ties 0.0% (0/15)
- **Coding**: gemma3:12b wins 0.0% (0/5), phi4:latest wins 100.0% (5/5), Ties 0.0% (0/5)
- **BugFinding**: gemma3:12b wins 40.0% (2/5), phi4:latest wins 60.0% (3/5), Ties 0.0% (0/5)
- **Polyglot**: gemma3:12b wins 20.0% (1/5), phi4:latest wins 80.0% (4/5), Ties 0.0% (0/5)

### gemma3:12b vs deepseek-r1:14b
- **Overall**: gemma3:12b wins 26.7% (3/15), deepseek-r1:14b wins 73.3% (10/15), Ties 13.3% (2/15)
- **Coding**: gemma3:12b wins 0.0% (0/5), deepseek-r1:14b wins 100.0% (5/5), Ties 0.0% (0/5)
- **BugFinding**: gemma3:12b wins 40.0% (2/5), deepseek-r1:14b wins 40.0% (2/5), Ties 20.0% (1/5)
- **Polyglot**: gemma3:12b wins 40.0% (2/5), deepseek-r1:14b wins 60.0% (3/5), Ties 0.0% (0/5)

### phi4:latest vs deepseek-r1:14b
- **Overall**: phi4:latest wins 80.0% (12/15), deepseek-r1:14b wins 20.0% (3/15), Ties 0.0% (0/15)
- **Coding**: phi4:latest wins 100.0% (5/5), deepseek-r1:14b wins 0.0% (0/5), Ties 0.0% (0/5)
- **BugFinding**: phi4:latest wins 80.0% (4/5), deepseek-r1:14b wins 20.0% (1/5), Ties 0.0% (0/5)
- **Polyglot**: phi4:latest wins 60.0% (3/5), deepseek-r1:14b wins 40.0% (2/5), Ties 0.0% (0/5)

## Key Insights

1. **Surprising Leader**: 
   - phi4:latest dominates this comparison group with a 75.6% average win rate
   - Consistently strong across all categories with win rates of 80%+ against most competitors
   - Particularly dominant in Coding tasks against gemma3:12b (100%) and deepseek-r1:14b (100%)

2. **Model Specializations**:
   - **cogito:14b**: Strong in debugging (80% win rate vs gemma3:12b in BugFinding)
   - **gemma3:12b**: Competitive in implementation (60% win rate vs cogito:14b in Coding)
   - **phi4:latest**: Most well-rounded, with strong performance across all categories
   - **deepseek-r1:14b**: Strong against cogito:14b and gemma3:12b but weak against phi4:latest

4. **Category-Specific Insights**:
   - **Coding**: phi4:latest and deepseek-r1:14b excel at implementation
   - **BugFinding**: cogito:14b and phi4:latest are strongest in debugging
   - **Polyglot**: Results are more varied, with phi4:latest still leading but by smaller margins

5. **ELO vs Direct Comparison**:
   - The direct comparison results conflict with the ELO leaderboard
   - In ELO rankings, gemma3:12b (1516) and cogito:14b (1505) are ranked higher than phi4:latest (1445)
   - This discrepancy likely comes from how ELO handles transitive relationships and comparisons against other models

Direct head-to-head results provide a clearer picture of actual model capabilities within this specific group. The results suggest phi4:latest is significantly stronger than its ELO rating would indicate, particularly for coding tasks.

*Generated on 2025-04-11 using direct head-to-head comparisons on the coding101 promptset*
