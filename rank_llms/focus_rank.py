"""Focus-based ranking system for LLM models based on direct comparisons with a focus model."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging
from collections import defaultdict

from rank_llms.compare import ComparisonResult, load_comparison_result
from rank_llms.leaderboard import find_all_comparison_files

logger = logging.getLogger("rank_llms")

class FocusRanking:
    """
    A ranking system that determines model rankings based on their performance
    against a selected focus model. Uses win ratio as the primary metric and can
    employ transitive relationships for models without direct comparisons.
    """
    
    def __init__(self, focus_model: str):
        """
        Initialize the focus ranking system.
        
        Args:
            focus_model: The model to use as the ranking focus
        """
        self.focus_model = focus_model
        self.comparisons = {}  # Dict to store comparison results
        self.models = set()  # Set of all models
        self.direct_ratios = {}  # Direct win ratios against focus model
        self.transitive_ratios = {}  # Transitive win ratios (estimated)
        self.graph = defaultdict(dict)  # Graph of win ratios between models
        
    def compute_rankings(self, promptset: str = "basic1", max_depth: int = 3) -> Dict[str, float]:
        """
        Compute rankings based on win ratios against the focus model.
        
        Args:
            promptset: Name of the promptset to use
            max_depth: Maximum depth for transitive relationships (1 = direct only)
        
        Returns:
            Dictionary of model rankings
        """
        self._load_all_comparisons(promptset)
        
        if self.focus_model not in self.models:
            logger.error(f"Focus model '{self.focus_model}' not found in any comparisons")
            return {}
        
        # Compute direct win ratios against focus model
        self._compute_direct_ratios()
        
        # Compute transitive win ratios if max_depth > 1
        if max_depth > 1:
            self._compute_transitive_ratios(max_depth)
        
        # Combine direct and transitive ratios
        all_ratios = {}
        all_ratios.update(self.direct_ratios)
        
        # Only add transitive ratios for models not directly compared
        for model, ratio in self.transitive_ratios.items():
            if model not in self.direct_ratios:
                all_ratios[model] = ratio
        
        # Add the focus model itself with a ratio of 1.0
        all_ratios[self.focus_model] = 1.0
        
        return all_ratios
    
    def _load_all_comparisons(self, promptset: str) -> None:
        """
        Load all comparison results for the given promptset.
        
        Args:
            promptset: Name of the promptset to use
        """
        # Find all comparison files
        comparison_files = find_all_comparison_files(promptset)
        logger.info(f"Found {len(comparison_files)} comparison files for promptset '{promptset}'")
        
        # Process each comparison file
        for file_path in comparison_files:
            try:
                # Extract model names from filename
                filename = file_path.name
                model_part = filename.replace(".json", "")
                file_models = model_part.split("__vs__")
                
                if len(file_models) != 2:
                    logger.warning(f"Invalid comparison filename format: {filename}")
                    continue
                
                model_a = file_models[0].replace("_", ":")
                model_b = file_models[1].replace("_", ":")
                
                # Add models to set
                self.models.add(model_a)
                self.models.add(model_b)
                
                # Load comparison result
                result = load_comparison_result(model_a, model_b, promptset=promptset)
                if result:
                    # Store comparison
                    comparison_key = tuple(sorted([model_a, model_b]))
                    self.comparisons[comparison_key] = result
                    
                    # Compute win ratio for this comparison and store in graph
                    total = result.overall_total
                    if total > 0:
                        # Calculate win rates (counting ties as half a win)
                        wins_a = result.overall_wins_a + 0.5 * result.overall_ties
                        wins_b = result.overall_wins_b + 0.5 * result.overall_ties
                        win_rate_a = wins_a / total
                        win_rate_b = wins_b / total
                        
                        # Store win ratios in graph
                        if win_rate_a > 0:  # Avoid division by zero
                            self.graph[model_b][model_a] = win_rate_b / win_rate_a
                        if win_rate_b > 0:  # Avoid division by zero
                            self.graph[model_a][model_b] = win_rate_a / win_rate_b
            except Exception as e:
                logger.error(f"Error processing comparison file {file_path}: {e}")
    
    def _compute_direct_ratios(self) -> None:
        """Compute direct win ratios against the focus model."""
        for model in self.models:
            if model == self.focus_model:
                continue
                
            comparison_key = tuple(sorted([self.focus_model, model]))
            if comparison_key in self.comparisons:
                result = self.comparisons[comparison_key]
                
                # Get models in correct order
                if result.model_a == self.focus_model:
                    focus_wins = result.overall_wins_a + 0.5 * result.overall_ties
                    other_wins = result.overall_wins_b + 0.5 * result.overall_ties
                else:
                    focus_wins = result.overall_wins_b + 0.5 * result.overall_ties
                    other_wins = result.overall_wins_a + 0.5 * result.overall_ties
                
                # Compute ratio
                if focus_wins > 0:
                    ratio = other_wins / focus_wins
                    self.direct_ratios[model] = ratio
                else:
                    # If focus model has no wins, assign a high ratio
                    self.direct_ratios[model] = float('inf')
    
    def _compute_transitive_ratios(self, max_depth: int) -> None:
        """
        Compute transitive win ratios for models not directly compared.
        
        Args:
            max_depth: Maximum path length for transitive relationships
        """
        # Find paths from focus model to all other models
        paths = self._find_paths(self.focus_model, max_depth)
        
        # Compute transitive ratios based on paths
        for model, path in paths.items():
            if model not in self.direct_ratios and model != self.focus_model:
                # Compute transitive ratio by multiplying ratios along path
                ratio = 1.0
                for i in range(len(path) - 1):
                    from_model = path[i]
                    to_model = path[i + 1]
                    ratio *= self.graph[from_model][to_model]
                
                self.transitive_ratios[model] = ratio
    
    def _find_paths(self, start: str, max_depth: int) -> Dict[str, List[str]]:
        """
        Find shortest paths from start model to all other models.
        
        Args:
            start: Starting model
            max_depth: Maximum path length
            
        Returns:
            Dictionary mapping models to their shortest path from start
        """
        # Use BFS to find shortest paths
        paths = {start: [start]}
        queue = [start]
        visited = {start}
        
        depth = 0
        while queue and depth < max_depth:
            depth += 1
            level_size = len(queue)
            
            for _ in range(level_size):
                current = queue.pop(0)
                
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        paths[neighbor] = paths[current] + [neighbor]
        
        return paths
    
    def get_ranking_table(self, ratios: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Create a ranking table from the win ratios.
        
        Args:
            ratios: Dictionary of model to win ratio
            
        Returns:
            List of (model, ratio, comparison_type) tuples sorted by ratio
        """
        # Create ranking table
        ranking = []
        
        for model, ratio in ratios.items():
            if model in self.direct_ratios:
                comparison_type = "direct"
            elif model == self.focus_model:
                comparison_type = "focus"
            else:
                comparison_type = "transitive"
            
            ranking.append((model, ratio, comparison_type))
        
        # Sort by ratio (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_raw_comparison_data(self) -> Dict[str, Dict]:
        """
        Get raw comparison data for models against the focus model.
        
        Returns:
            Dictionary mapping models to their comparison data
        """
        result = {}
        
        for model in self.models:
            if model == self.focus_model:
                continue
                
            comparison_key = tuple(sorted([self.focus_model, model]))
            if comparison_key in self.comparisons:
                comparison = self.comparisons[comparison_key]
                
                # Get models in correct order
                if comparison.model_a == self.focus_model:
                    focus_wins = comparison.overall_wins_a
                    other_wins = comparison.overall_wins_b
                    ties = comparison.overall_ties
                    total = comparison.overall_total
                else:
                    focus_wins = comparison.overall_wins_b
                    other_wins = comparison.overall_wins_a
                    ties = comparison.overall_ties
                    total = comparison.overall_total
                
                # Category results
                category_data = {}
                for category, cat_result in comparison.category_results.items():
                    if cat_result.model_a == self.focus_model:
                        cat_focus_wins = cat_result.wins_a
                        cat_other_wins = cat_result.wins_b
                        cat_ties = cat_result.ties
                    else:
                        cat_focus_wins = cat_result.wins_b
                        cat_other_wins = cat_result.wins_a
                        cat_ties = cat_result.ties
                    
                    category_data[category] = {
                        "focus_wins": cat_focus_wins,
                        "other_wins": cat_other_wins,
                        "ties": cat_ties,
                        "total": cat_focus_wins + cat_other_wins + cat_ties
                    }
                
                result[model] = {
                    "focus_wins": focus_wins,
                    "other_wins": other_wins,
                    "ties": ties,
                    "total": total,
                    "categories": category_data
                }
        
        return result


def format_focus_ranking_markdown(
    focus_model: str, 
    ranking: List[Tuple[str, float, str]], 
    comparison_data: Dict[str, Dict],
    promptset: str = "basic1",
    max_depth: int = 3
) -> str:
    """
    Format focus ranking results as markdown.
    
    Args:
        focus_model: The focus model
        ranking: List of (model, ratio, comparison_type) tuples
        comparison_data: Raw comparison data
        promptset: Name of the promptset used
        max_depth: Maximum depth used for transitive relationships
        
    Returns:
        Markdown-formatted ranking table and comparison details
    """
    # Create markdown output
    markdown = f"# Focus-Based Model Rankings: {focus_model}\n\n"
    markdown += f"This ranking is based on win ratios against the focus model '{focus_model}' using the '{promptset}' promptset.\n"
    
    if max_depth > 1:
        markdown += f"Transitive relationships up to depth {max_depth} are included.\n\n"
    else:
        markdown += "Only direct comparisons are included.\n\n"
    
    # Rankings table
    markdown += "## Model Rankings\n\n"
    markdown += "| Rank | Model | Win Ratio | Comparison Type |\n"
    markdown += "|------|-------|-----------|----------------|\n"
    
    for i, (model, ratio, comp_type) in enumerate(ranking, 1):
        # Format ratio
        if ratio == float('inf'):
            ratio_str = "∞"
        else:
            ratio_str = f"{ratio:.2f}"
        
        # Add rank indicators
        if model == focus_model:
            rank_indicator = "="
        elif ratio > 1.0:
            rank_indicator = "▲"  # Higher than focus
        else:
            rank_indicator = "▼"  # Lower than focus
        
        markdown += f"| {i} | {model} | {ratio_str} {rank_indicator} | {comp_type} |\n"
    
    # Add explanation
    markdown += "\n**Win Ratio Explanation:**\n"
    markdown += "- **> 1.0**: Model outperforms the focus model\n"
    markdown += "- **= 1.0**: Equal performance with focus model\n"
    markdown += "- **< 1.0**: Model underperforms the focus model\n"
    markdown += "- **∞**: Focus model has not won any comparisons against this model\n\n"
    
    # Direct comparison details
    if comparison_data:
        markdown += "## Direct Comparison Details\n\n"
        
        for model, data in comparison_data.items():
            focus_wins = data["focus_wins"]
            other_wins = data["other_wins"]
            ties = data["ties"]
            total = data["total"]
            
            # Calculate win rates
            focus_win_rate = (focus_wins + 0.5 * ties) / total if total > 0 else 0
            other_win_rate = (other_wins + 0.5 * ties) / total if total > 0 else 0
            
            # Calculate ratio
            if focus_win_rate > 0:
                ratio = other_win_rate / focus_win_rate
                ratio_str = f"{ratio:.2f}"
            else:
                ratio_str = "∞"
            
            markdown += f"### {focus_model} vs {model} (Win Ratio: {ratio_str})\n\n"
            markdown += f"- **Overall**: {focus_model} wins {focus_wins}/{total} ({focus_win_rate:.1%}), "
            markdown += f"{model} wins {other_wins}/{total} ({other_win_rate:.1%}), "
            markdown += f"Ties {ties}/{total} ({ties/total:.1%})\n\n"
            
            # Category breakdown
            if "categories" in data:
                markdown += "**Category Breakdown:**\n\n"
                
                for category, cat_data in data["categories"].items():
                    cat_focus_wins = cat_data["focus_wins"]
                    cat_other_wins = cat_data["other_wins"]
                    cat_ties = cat_data["ties"]
                    cat_total = cat_data["total"]
                    
                    if cat_total > 0:
                        cat_focus_win_rate = (cat_focus_wins + 0.5 * cat_ties) / cat_total
                        cat_other_win_rate = (cat_other_wins + 0.5 * cat_ties) / cat_total
                        
                        markdown += f"- **{category}**: {focus_model} wins {cat_focus_wins}/{cat_total} ({cat_focus_win_rate:.1%}), "
                        markdown += f"{model} wins {cat_other_wins}/{cat_total} ({cat_other_win_rate:.1%}), "
                        markdown += f"Ties {cat_ties}/{cat_total} ({cat_ties/cat_total:.1%})\n"
                
                markdown += "\n"
    
    # Transitive relationships
    if max_depth > 1:
        transitive_models = [m for m, _, t in ranking if t == "transitive"]
        if transitive_models:
            markdown += "## Transitive Relationships\n\n"
            markdown += "These models have no direct comparisons with the focus model. "
            markdown += "Their rankings are estimated based on transitive relationships.\n\n"
            
            for model in transitive_models:
                for _, (m, r, _) in enumerate(ranking):
                    if m == model:
                        if r == float('inf'):
                            r_str = "∞"
                        else:
                            r_str = f"{r:.2f}"
                        markdown += f"- **{model}**: Estimated win ratio {r_str}\n"
            
            markdown += "\n"
    
    # Add generation information
    markdown += f"*Generated on 2025-04-11 using the focus ranking method on the {promptset} promptset*\n"
    
    return markdown