"""Direct comparison model for ranking LLM models based on head-to-head results."""

import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import logging
from pathlib import Path

from rank_llms.compare import ComparisonResult, load_comparison_result
from rank_llms.leaderboard import find_all_comparison_files

logger = logging.getLogger("rank_llms")

class DirectComparisonRanking:
    """
    A simple ranking system based on direct head-to-head comparison results.
    Only uses actual comparison data without inferring transitive relationships.
    """
    
    def __init__(self):
        """Initialize the direct comparison ranking system."""
        self.win_matrix = None  # Matrix of wins between pairs of models
        self.probability_matrix = None  # Matrix of win probabilities
        self.models = []  # List of models in the analysis
        self.match_counts = None  # Matrix of match counts between pairs of models
        self.missing_comparisons = []  # List of missing comparisons
        
    def compute_rankings(self, models: List[str], promptset: str = "basic1") -> bool:
        """
        Compute rankings based on direct comparison results.
        
        Args:
            models: List of models to include in the analysis
            promptset: Name of the promptset to use
        
        Returns:
            True if all comparisons are available, False otherwise
        """
        self.models = models
        self.win_matrix = pd.DataFrame(0, index=models, columns=models)
        self.match_counts = pd.DataFrame(0, index=models, columns=models)
        self.missing_comparisons = []
        
        # Find all comparison files for this promptset
        comparison_files = find_all_comparison_files(promptset)
        logger.info(f"Found {len(comparison_files)} comparison files for promptset '{promptset}'")
        
        # Load relevant comparison results
        for model_a in models:
            for model_b in models:
                if model_a != model_b:
                    result = load_comparison_result(model_a, model_b, promptset=promptset)
                    if result:
                        # Update win matrix
                        self.win_matrix.loc[model_a, model_b] = result.overall_wins_a
                        self.win_matrix.loc[model_b, model_a] = result.overall_wins_b
                        
                        # Update match counts (wins + ties)
                        total_comparisons = result.overall_wins_a + result.overall_wins_b + result.overall_ties
                        self.match_counts.loc[model_a, model_b] = total_comparisons
                        self.match_counts.loc[model_b, model_a] = total_comparisons
                    else:
                        # Record missing comparison
                        self.missing_comparisons.append((model_a, model_b))
        
        # Check if all comparisons are available
        if self.missing_comparisons:
            return False
        
        # Calculate win probabilities
        self.probability_matrix = pd.DataFrame(index=models, columns=models)
        for model_a in models:
            for model_b in models:
                if model_a == model_b:
                    self.probability_matrix.loc[model_a, model_b] = 0.5  # Same model, 50% probability
                else:
                    total_matches = self.match_counts.loc[model_a, model_b]
                    if total_matches > 0:
                        wins_a = self.win_matrix.loc[model_a, model_b]
                        # Counting ties as half a win for each model
                        ties = total_matches - wins_a - self.win_matrix.loc[model_b, model_a]
                        win_prob = (wins_a + 0.5 * ties) / total_matches
                        self.probability_matrix.loc[model_a, model_b] = win_prob
                    else:
                        self.probability_matrix.loc[model_a, model_b] = None
        
        return True
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get a sorted list of (model, score) tuples, from highest to lowest score.
        Score is the average win probability against all other models.
        
        Returns:
            List of tuples (model_name, score)
        """
        if self.probability_matrix is None:
            raise ValueError("Rankings have not been computed yet. Call compute_rankings() first.")
        
        scores = {}
        for model in self.models:
            # Calculate average win probability against all other models
            win_probs = []
            for opponent in self.models:
                if model != opponent:
                    win_probs.append(self.probability_matrix.loc[model, opponent])
            scores[model] = sum(win_probs) / len(win_probs) if win_probs else 0
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_missing_comparison_commands(self, promptset: str = "basic1") -> List[str]:
        """
        Get list of commands to run missing comparisons.
        
        Args:
            promptset: Name of the promptset to use
            
        Returns:
            List of commands to run missing comparisons
        """
        commands = []
        for model_a, model_b in self.missing_comparisons:
            commands.append(f"rank-llms compare {model_a} {model_b} --promptset {promptset}")
        return commands


def format_comparison_results_markdown(ranking: DirectComparisonRanking) -> str:
    """
    Format the comparison results as a markdown table.
    
    Args:
        ranking: DirectComparisonRanking instance with computed rankings
    
    Returns:
        Markdown-formatted table of win probabilities and match results
    """
    prob_matrix = ranking.probability_matrix
    rankings = ranking.get_rankings()
    ranked_models = [model for model, _ in rankings]
    
    # Create markdown table
    markdown = "# Direct Comparison Results\n\n"
    
    # Overall rankings
    markdown += "## Overall Rankings\n\n"
    markdown += "| Rank | Model | Average Win Rate |\n"
    markdown += "|------|-------|----------------|\n"
    
    for i, (model, score) in enumerate(rankings, 1):
        markdown += f"| {i} | {model} | {score:.3f} |\n"
    
    # Win probability matrix
    markdown += "\n## Win Probability Matrix\n\n"
    markdown += "Probability of row model beating column model (based on head-to-head results):\n\n"
    
    # Header row
    markdown += "| Model |"
    for model in ranked_models:
        markdown += f" {model} |"
    markdown += "\n"
    
    # Separator row
    markdown += "|-------|"
    for _ in ranked_models:
        markdown += "-------|"
    markdown += "\n"
    
    # Data rows
    for row_model in ranked_models:
        markdown += f"| **{row_model}** |"
        for col_model in ranked_models:
            if row_model == col_model:
                markdown += " - |"
            else:
                prob = prob_matrix.loc[row_model, col_model]
                markdown += f" {prob:.3f} |" if pd.notnull(prob) else " N/A |"
        markdown += "\n"
    
    # Head-to-head results
    markdown += "\n## Detailed Head-to-Head Results\n\n"
    
    for i, model_a in enumerate(ranked_models):
        for model_b in ranked_models[i+1:]:
            total_matches = ranking.match_counts.loc[model_a, model_b]
            if total_matches > 0:
                wins_a = ranking.win_matrix.loc[model_a, model_b]
                wins_b = ranking.win_matrix.loc[model_b, model_a]
                ties = total_matches - wins_a - wins_b
                
                win_rate_a = (wins_a + 0.5 * ties) / total_matches
                win_rate_b = (wins_b + 0.5 * ties) / total_matches
                
                markdown += f"### {model_a} vs {model_b}\n\n"
                markdown += f"- **Overall**: {model_a} wins {wins_a}/{total_matches} ({win_rate_a:.1%}), {model_b} wins {wins_b}/{total_matches} ({win_rate_b:.1%}), Ties {ties}/{total_matches} ({ties/total_matches:.1%})\n\n"
    
    return markdown


def format_missing_comparisons_markdown(ranking: DirectComparisonRanking, promptset: str = "basic1") -> str:
    """
    Format missing comparisons as markdown with commands to run.
    
    Args:
        ranking: DirectComparisonRanking instance
        promptset: Name of the promptset to use
    
    Returns:
        Markdown-formatted list of missing comparisons
    """
    markdown = "# Missing Comparisons\n\n"
    markdown += "The following comparisons are needed for a complete analysis:\n\n"
    
    for i, (model_a, model_b) in enumerate(ranking.missing_comparisons, 1):
        command = f"rank-llms compare {model_a} {model_b} --promptset {promptset}"
        markdown += f"{i}. Compare **{model_a}** with **{model_b}**:\n"
        markdown += f"   ```\n   {command}\n   ```\n\n"
    
    return markdown