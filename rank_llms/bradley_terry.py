"""Bradley-Terry model for ranking LLM models based on comparative evaluations."""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import json
from pathlib import Path
import logging
from collections import defaultdict

from rank_llms.compare import ComparisonResult, load_comparison_result
from rank_llms.leaderboard import find_all_comparison_files

logger = logging.getLogger("rank_llms")

class BradleyTerryModel:
    """
    Implements a Bradley-Terry model for ranking LLM models based on head-to-head comparisons.
    The Bradley-Terry model estimates the probability of one model beating another based on
    their underlying "strength" parameters.
    """
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 1e-6):
        """
        Initialize the Bradley-Terry model.
        
        Args:
            max_iterations: Maximum number of iterations for MLE estimation
            convergence_threshold: Threshold for convergence in MLE estimation
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.strengths = {}  # Model strengths
        self.win_matrix = None  # Matrix of wins between pairs of models
        self.models = []  # List of models in the analysis
        
    def fit(self, win_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Fit the Bradley-Terry model to the given win matrix.
        
        Args:
            win_matrix: DataFrame where rows and columns are models,
                       and values are the number of wins for row model against column model
                       
        Returns:
            Dictionary mapping model names to strength parameters
        """
        self.win_matrix = win_matrix
        self.models = list(win_matrix.index)
        n_models = len(self.models)
        
        # Initialize strengths with equal values
        strengths = np.ones(n_models) / n_models
        
        # Maximum likelihood estimation using iterative method
        for _ in range(self.max_iterations):
            new_strengths = np.zeros(n_models)
            
            for i in range(n_models):
                # Total number of comparisons involving model i
                total_comparisons = 0
                for j in range(n_models):
                    if i != j:
                        total_comparisons += win_matrix.iloc[i, j] + win_matrix.iloc[j, i]
                
                if total_comparisons == 0:
                    # No comparisons for this model, keep strength unchanged
                    new_strengths[i] = strengths[i]
                    continue
                
                # Total number of wins for model i
                wins_i = win_matrix.iloc[i, :].sum()
                
                # Sum of adjusted opponent strengths
                denominator_sum = 0
                for j in range(n_models):
                    if i != j:
                        pair_matches = win_matrix.iloc[i, j] + win_matrix.iloc[j, i]
                        if pair_matches > 0:
                            denominator_sum += pair_matches * strengths[j] / (strengths[i] + strengths[j])
                
                # Update strength using MLE formula
                if denominator_sum > 0:
                    new_strengths[i] = wins_i / denominator_sum
                else:
                    new_strengths[i] = strengths[i]
            
            # Normalize to sum to 1
            new_strengths = new_strengths / new_strengths.sum()
            
            # Check for convergence
            if np.max(np.abs(new_strengths - strengths)) < self.convergence_threshold:
                break
                
            strengths = new_strengths
        
        # Store the results
        self.strengths = {model: float(strength) for model, strength in zip(self.models, strengths)}
        return self.strengths
    
    def probability_matrix(self) -> pd.DataFrame:
        """
        Calculate the probability matrix of one model beating another.
        
        Returns:
            DataFrame where entry (i,j) is the probability of model i beating model j
        """
        if not self.strengths:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        n_models = len(self.models)
        prob_matrix = pd.DataFrame(index=self.models, columns=self.models)
        
        for i, model_i in enumerate(self.models):
            for j, model_j in enumerate(self.models):
                if i == j:
                    prob_matrix.loc[model_i, model_j] = 0.5  # Same model, 50% probability
                else:
                    p_i = self.strengths[model_i]
                    p_j = self.strengths[model_j]
                    prob_matrix.loc[model_i, model_j] = p_i / (p_i + p_j)
        
        return prob_matrix
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get a sorted list of (model, strength) tuples, from highest to lowest strength.
        
        Returns:
            List of tuples (model_name, strength_parameter)
        """
        if not self.strengths:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return sorted(self.strengths.items(), key=lambda x: x[1], reverse=True)


def build_win_matrix(models: List[str], promptset: str = "basic1") -> pd.DataFrame:
    """
    Build a win matrix from the comparison results in the test archive.
    
    Args:
        models: List of models to include in the analysis
        promptset: Name of the promptset to use
    
    Returns:
        DataFrame where entry (i,j) is the number of times model i beat model j
    """
    # Initialize win matrix
    win_matrix = pd.DataFrame(0, index=models, columns=models)
    
    # Find all comparison files for this promptset
    comparison_files = find_all_comparison_files(promptset)
    logger.info(f"Found {len(comparison_files)} comparison files for promptset '{promptset}'")
    
    # Load relevant comparison results
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
            
            # Skip if either model is not in our subset
            if model_a not in models or model_b not in models:
                continue
            
            # Load the comparison result
            result = load_comparison_result(model_a, model_b, promptset=promptset)
            if result:
                # Update win matrix
                win_matrix.loc[model_a, model_b] = result.overall_wins_a
                win_matrix.loc[model_b, model_a] = result.overall_wins_b
        except Exception as e:
            logger.error(f"Error processing comparison file {file_path}: {e}")
    
    return win_matrix


def generate_bradley_terry_rankings(models: List[str], promptset: str = "basic1") -> BradleyTerryModel:
    """
    Generate Bradley-Terry model rankings for a subset of models.
    
    Args:
        models: List of models to include in the analysis
        promptset: Name of the promptset to use
    
    Returns:
        Fitted BradleyTerryModel instance
    """
    # Build win matrix
    win_matrix = build_win_matrix(models, promptset)
    
    # Fit Bradley-Terry model
    bt_model = BradleyTerryModel()
    bt_model.fit(win_matrix)
    
    return bt_model


def format_probability_matrix_markdown(bt_model: BradleyTerryModel) -> str:
    """
    Format the probability matrix as a markdown table.
    
    Args:
        bt_model: Fitted BradleyTerryModel instance
    
    Returns:
        Markdown-formatted table of win probabilities
    """
    prob_matrix = bt_model.probability_matrix()
    rankings = bt_model.get_rankings()
    ranked_models = [model for model, _ in rankings]
    
    # Create markdown table
    markdown = "# Bradley-Terry Model Win Probability Matrix\n\n"
    markdown += "Estimated probability of row model beating column model:\n\n"
    
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
                markdown += f" {prob:.3f} |"
        markdown += "\n"
    
    # Add model strengths
    markdown += "\n## Model Strength Parameters\n\n"
    markdown += "| Rank | Model | Strength |\n"
    markdown += "|------|-------|----------|\n"
    
    for i, (model, strength) in enumerate(rankings, 1):
        # Convert strength to a more readable scale
        scaled_strength = strength * 100
        markdown += f"| {i} | {model} | {scaled_strength:.2f} |\n"
    
    return markdown