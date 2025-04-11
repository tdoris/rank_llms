# Coding-Focused LLM Models Leaderboard

**Generated on:** 2025-04-11

This leaderboard ranks models based on their performance in coding-related tasks using the `coding101` promptset.

## Overall Rankings

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1599 |
| 2 | cogito:32b | 1558 |
| 3 | gemma3:12b | 1498 |
| 4 | cogito:14b | 1481 |
| 5 | mistral-small3.1:24b | 1455 |
| 6 | deepseek-r1:32b | 1454 |
| 7 | phi4:latest | 1441 |
| 8 | qwen2.5-coder:32b | 1430 |
| 9 | llama3.3:70b | 1359 |
| 10 | deepseek-r1:14b | 1352 |
| 11 | deepcoder:latest | 1286 |
| 12 | gemma3:4b | 1226 |
| 13 | llama3.1:8b | 1060 |

## Category Rankings

### Coding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | gemma3:27b | 1779 |
| 2 | cogito:32b | 1595 |
| 3 | gemma3:12b | 1539 |
| 4 | phi4:latest | 1444 |
| 5 | cogito:14b | 1426 |
| 6 | deepseek-r1:32b | 1403 |
| 7 | mistral-small3.1:24b | 1401 |
| 8 | gemma3:4b | 1393 |
| 9 | qwen2.5-coder:32b | 1331 |
| 10 | deepseek-r1:14b | 1310 |
| 11 | llama3.3:70b | 1257 |
| 12 | deepcoder:latest | 1219 |
| 13 | llama3.1:8b | 1102 |

### Bug Finding

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | cogito:32b | 1608 |
| 2 | cogito:14b | 1535 |
| 3 | deepseek-r1:32b | 1487 |
| 4 | llama3.3:70b | 1476 |
| 5 | mistral-small3.1:24b | 1470 |
| 6 | gemma3:27b | 1441 |
| 7 | phi4:latest | 1440 |
| 8 | gemma3:12b | 1437 |
| 9 | deepseek-r1:14b | 1387 |
| 10 | qwen2.5-coder:32b | 1406 |
| 11 | deepcoder:latest | 1347 |
| 12 | gemma3:4b | 1089 |
| 13 | llama3.1:8b | 1077 |

### Polyglot

| Rank | Model | ELO Rating |
|------|-------|------------|
| 1 | qwen2.5-coder:32b | 1528 |
| 2 | gemma3:12b | 1520 |
| 3 | mistral-small3.1:24b | 1501 |
| 4 | cogito:32b | 1492 |
| 5 | cogito:14b | 1484 |
| 6 | deepseek-r1:32b | 1473 |
| 7 | gemma3:27b | 1622 |
| 8 | phi4:latest | 1442 |
| 9 | deepseek-r1:14b | 1359 |
| 10 | llama3.3:70b | 1339 |
| 11 | deepcoder:latest | 1294 |
| 12 | gemma3:4b | 1157 |
| 13 | llama3.1:8b | 988 |

## Analysis

### Tier Analysis

Based on performance, models can be grouped into the following tiers:

**Tier 1 (Elite):**
- gemma3:27b (1599) - Exceptional performance across all categories, dominates in coding tasks
- cogito:32b (1558) - Excellent overall, exceptional in bug finding

**Tier 2 (Strong Performers):**
- gemma3:12b (1498) - Strong balanced performance across categories
- cogito:14b (1481) - Excellent bug finding abilities
- mistral-small3.1:24b (1455) - Well-rounded performance
- deepseek-r1:32b (1454) - Solid general-purpose coding model

**Tier 3 (Competent):**
- phi4:latest (1441) - Good balanced performance
- qwen2.5-coder:32b (1430) - Strong in polyglot tasks, weaker in coding implementation

**Tier 4 (Average):**
- llama3.3:70b (1359) - Decent bug finding, weaker in coding
- deepseek-r1:14b (1352) - Middle-of-the-pack performance
- deepcoder:latest (1286) - Better at bug finding than implementation

**Tier 5 (Entry Level):**
- gemma3:4b (1226) - Impressive for its size, lacks consistent performance
- llama3.1:8b (1060) - Struggles across all categories

### Specialization Analysis

Some models show distinct specialization patterns:

- **Balanced Models:** gemma3:12b, mistral-small3.1:24b
- **Implementation Specialists:** gemma3:27b (exceptional), cogito:32b
- **Bug Finding Specialists:** cogito:32b, cogito:14b
- **Polyglot Specialists:** qwen2.5-coder:32b, gemma3:12b
- **Size-Performance Outliers:** gemma3:4b (punches above its weight class)

### Size-Performance Analysis

Model size correlates with performance, but with interesting exceptions:

- **Large Models (27B-70B):** Generally strong but with significant variation (gemma3:27b vs llama3.3:70b)
- **Mid-Size Models (12B-24B):** Excellent performance/size ratio, especially gemma3:12b
- **Smaller Models (<8B):** Struggle with complex coding tasks, but gemma3:4b shows surprising competence

## Conclusion

The coding benchmark reveals that model architecture and training approach matter as much as size. Gemma and Cogito models show particularly strong performance, with specialized capabilities emerging across different coding tasks. For optimal results, consider using gemma3:27b for general coding tasks, cogito:32b for debugging, and qwen2.5-coder:32b for polyglot programming.