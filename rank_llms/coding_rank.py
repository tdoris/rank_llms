#!/usr/bin/env python3
"""
Script to generate coding-specific rankings and win probability matrix from head-to-head comparison results.
Focuses only on the "Coding" category within the coding101 promptset.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
from collections import defaultdict
import re

from rank_llms.stats import wilson_interval, intervals_overlap


class CodingRankAnalyzer:
    """Analyzes coding-specific performance based on head-to-head comparisons."""

    def __init__(
        self,
        test_archive_dir: str = "test_archive",
        promptset: str = "coding101",
        category: str = "Coding",
    ):
        """
        Initialize the analyzer.

        Args:
            test_archive_dir: Directory containing test archives
            promptset: Name of the promptset whose comparisons to analyze
            category: Name of the category to rank on (case-insensitive)
        """
        self.test_archive_dir = test_archive_dir
        self.promptset = promptset
        self.category = category
        self.comparison_dir = os.path.join(test_archive_dir, promptset, "comparisons")
        if not os.path.exists(self.comparison_dir):
            raise ValueError(f"Comparison directory not found: {self.comparison_dir}")

    def parse_comparison_file(self, file_path: str) -> Dict:
        """
        Parse a comparison file to extract Coding category results.
        
        Args:
            file_path: Path to the comparison JSON file
            
        Returns:
            Dictionary with comparison results
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # First try standard JSON parsing
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract the JSON portion using regex
            json_pattern = re.search(r'(\{[\s\S]*\})', content)
            if json_pattern:
                try:
                    data = json.loads(json_pattern.group(1))
                except json.JSONDecodeError:
                    # If that still fails, return empty results
                    return {"model_a": "", "model_b": "", "coding_wins_a": 0, "coding_wins_b": 0, "coding_ties": 0}
            else:
                return {"model_a": "", "model_b": "", "coding_wins_a": 0, "coding_wins_b": 0, "coding_ties": 0}
        
        # Extract models from the filename or data
        filename = os.path.basename(file_path)
        models = filename.replace('.json', '').split('__vs__')
        
        if len(models) != 2:
            # Try to get models from data if filename parsing fails
            model_a = data.get('model_a', '')
            model_b = data.get('model_b', '')
        else:
            # Convert underscores back to colons for model names in the format model_name_parameter
            model_a = models[0]
            model_b = models[1]
            
            # Handle special format transformations
            for model_var in ['model_a', 'model_b']:
                model_name = locals()[model_var]
                
                # Replace specific patterns
                if 'deepseek-r1_' in model_name:
                    locals()[model_var] = model_name.replace('deepseek-r1_', 'deepseek-r1:')
                elif 'phi4_' in model_name:
                    locals()[model_var] = model_name.replace('phi4_', 'phi4:')
                elif 'qwen2.5-coder_' in model_name:
                    locals()[model_var] = model_name.replace('qwen2.5-coder_', 'qwen2.5-coder:')
                elif 'cogito_' in model_name:
                    locals()[model_var] = model_name.replace('cogito_', 'cogito:')
                elif 'gemma3_' in model_name:
                    locals()[model_var] = model_name.replace('gemma3_', 'gemma3:')
                # Add more specific replacements as needed
                
            # If no specific pattern matched, try a more general approach
            if '_' in model_a and ':' not in model_a:
                # Find the last underscore and replace it with a colon
                last_underscore = model_a.rindex('_')
                model_a = model_a[:last_underscore] + ':' + model_a[last_underscore+1:]
                
            if '_' in model_b and ':' not in model_b:
                # Find the last underscore and replace it with a colon
                last_underscore = model_b.rindex('_')
                model_b = model_b[:last_underscore] + ':' + model_b[last_underscore+1:]
        
        # Extract Coding category results
        coding_wins_a = 0
        coding_wins_b = 0
        coding_ties = 0
        
        # Handle the special case in the coding101 promptset where 'a' and 'b' are used instead of actual model names
        if 'comparisons' in data:
            for comparison in data['comparisons']:
                # Check if this is a Coding category comparison
                category = comparison.get('category', '')
                winner = comparison.get('winner', '')
                
                if category.lower() == self.category.lower():
                    # In the coding101 comparisons, winner is often 'a' or 'b' instead of the model name
                    if winner == 'a':
                        coding_wins_a += 1
                    elif winner == 'b':
                        coding_wins_b += 1
                    elif winner == 'tie':
                        coding_ties += 1
                    elif winner == model_a:
                        coding_wins_a += 1
                    elif winner == model_b:
                        coding_wins_b += 1
                    else:
                        coding_ties += 1
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "coding_wins_a": coding_wins_a,
            "coding_wins_b": coding_wins_b,
            "coding_ties": coding_ties
        }

    def get_comparison_files(self, models: List[str]) -> List[str]:
        """
        Get all comparison files for the given models.
        
        Args:
            models: List of model names
            
        Returns:
            List of file paths
        """
        comparison_files = []
        
        for filename in os.listdir(self.comparison_dir):
            if not filename.endswith('.json'):
                continue
                
            # Replace colons with underscores to match filename format
            model_patterns = [model.replace(':', '_') for model in models]
            
            # Check if the file contains a comparison between any of our target models
            matches = sum(1 for pattern in model_patterns if pattern in filename)
            if matches >= 1:  # File involves at least one of our target models
                comparison_files.append(os.path.join(self.comparison_dir, filename))
                
        return comparison_files

    def build_win_matrix(
        self, models: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, int]]]:
        """
        Build a win matrix for the given models based on Coding category results.

        Args:
            models: List of model names

        Returns:
            Tuple of (win_matrix, win_rates, win_counts) where win_matrix is a
            DataFrame, win_rates maps model names to their average win rate, and
            win_counts maps model names to pooled {"wins", "comparisons"} totals
            used for confidence intervals.
        """
        # Initialize win matrix with NaN values
        win_matrix = pd.DataFrame(np.nan, index=models, columns=models)

        # Set diagonal to NaN (no self-comparisons)
        for model in models:
            win_matrix.loc[model, model] = float('nan')

        # Get all comparison files
        comparison_files = self.get_comparison_files(models)

        # Pooled raw counts per model (across all opponents) for CIs
        win_counts = {model: {"wins": 0, "comparisons": 0} for model in models}

        # Extract Coding category results from each file
        for file_path in comparison_files:
            results = self.parse_comparison_file(file_path)

            model_a = results["model_a"]
            model_b = results["model_b"]

            # Skip if either model is not in our target list
            if model_a not in models or model_b not in models:
                continue

            # Calculate win probabilities
            total_comparisons = results["coding_wins_a"] + results["coding_wins_b"] + results["coding_ties"]
            if total_comparisons > 0:
                a_beats_b = results["coding_wins_a"] / total_comparisons
                b_beats_a = results["coding_wins_b"] / total_comparisons

                win_matrix.loc[model_a, model_b] = a_beats_b
                win_matrix.loc[model_b, model_a] = b_beats_a

                # Accumulate pooled counts (ties count toward the denominator,
                # matching the win-rate semantics used elsewhere)
                win_counts[model_a]["wins"] += results["coding_wins_a"]
                win_counts[model_a]["comparisons"] += total_comparisons
                win_counts[model_b]["wins"] += results["coding_wins_b"]
                win_counts[model_b]["comparisons"] += total_comparisons

        # Calculate win rates for each model
        win_rates = {}
        for model in models:
            # Calculate average win rate against all other models
            model_rates = win_matrix.loc[model, :].dropna()
            if len(model_rates) > 0:
                win_rates[model] = model_rates.mean()
            else:
                win_rates[model] = float('nan')

        return win_matrix, win_rates, win_counts

    def generate_rankings(self, models: List[str]) -> Dict:
        """
        Generate rankings for the given models.
        
        Args:
            models: List of model names
            
        Returns:
            Dictionary with win matrix and rankings
        """
        win_matrix, win_rates, win_counts = self.build_win_matrix(models)

        # Separate models that have comparison data from those that don't.
        # NaN win rates cannot be sorted meaningfully (NaN compares False
        # against every value), so ranking them would scramble the order and
        # produce non-contiguous rank numbers downstream.
        ranked = {model: rate for model, rate in win_rates.items() if not np.isnan(rate)}
        no_data_models = [model for model, rate in win_rates.items() if np.isnan(rate)]

        # Sort the models that have data by win rate (highest first).
        sorted_models = sorted(ranked.items(), key=lambda x: x[1], reverse=True)

        # Wilson confidence interval + sample size per ranked model, from pooled
        # raw counts. Lets callers report uncertainty and flag models whose
        # intervals overlap (not statistically distinguishable).
        confidence = {}
        for model, _rate in sorted_models:
            counts = win_counts.get(model, {"wins": 0, "comparisons": 0})
            n = counts["comparisons"]
            confidence[model] = {
                "wins": counts["wins"],
                "comparisons": n,
                "interval": wilson_interval(counts["wins"], n),
            }

        return {
            "win_matrix": win_matrix,
            "rankings": sorted_models,
            "no_data_models": no_data_models,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }

    def generate_markdown(self, results: Dict, output_file: str = None) -> str:
        """
        Generate a markdown report from the results.
        
        Args:
            results: Dictionary with win matrix and rankings
            output_file: Path to output file (if None, returns the markdown as a string)
            
        Returns:
            Generated markdown as a string
        """
        win_matrix = results["win_matrix"]
        rankings = results["rankings"]
        timestamp = results["timestamp"]
        
        markdown = f"# {self.category}-Specific Performance Analysis\n\n"
        markdown += f"This analysis shows the performance of models specifically on the {self.category} category tasks within the {self.promptset} promptset, "
        markdown += f"based on actual head-to-head test results.\n\n"

        # Rankings table
        markdown += f"## {self.category}-Only Rankings\n\n"

        confidence = results.get("confidence", {})

        if not rankings:
            markdown += f"*No {self.category}-category comparison data found for the requested models.*\n"
        else:
            markdown += f"| Rank | Model | {self.category} Win Rate | N | 95% CI |\n"
            markdown += f"|------|-------|----------------|---|--------|\n"

            # rankings only contains models with data, so ranks are contiguous.
            for i, (model, win_rate) in enumerate(rankings):
                info = confidence.get(model, {})
                n = info.get("comparisons", 0)
                interval = info.get("interval")
                ci_text = f"{interval[0]:.3f}–{interval[1]:.3f}" if interval else "N/A"
                markdown += f"| {i+1} | {model} | {win_rate:.3f} | {n} | {ci_text} |\n"

            # Flag adjacent models whose confidence intervals overlap — their
            # ordering is not statistically distinguishable at 95%.
            overlaps = []
            for (m1, _), (m2, _) in zip(rankings, rankings[1:]):
                if intervals_overlap(
                    confidence.get(m1, {}).get("interval"),
                    confidence.get(m2, {}).get("interval"),
                ):
                    overlaps.append((m1, m2))
            if overlaps:
                markdown += "\n*Not statistically distinguishable (overlapping 95% CIs): "
                markdown += "; ".join(f"{a} vs {b}" for a, b in overlaps)
                markdown += ".*\n"

        # Win probability matrix
        markdown += f"\n## {self.category} Win Probability Matrix\n\n"
        markdown += f"Probability of row model beating column model in {self.category} tasks only:\n\n"

        # Models that have data (all entries in rankings already qualify)
        valid_models = [model for model, rate in rankings]
        
        # Header row
        markdown += f"| Model | {' | '.join(valid_models)} |\n"
        markdown += f"|-------|" + "|".join(["-" * 7] * len(valid_models)) + "|\n"
        
        # Data rows
        for row_model in valid_models:
            row_data = []
            for col_model in valid_models:
                if row_model == col_model:
                    row_data.append("-")
                else:
                    value = win_matrix.loc[row_model, col_model]
                    row_data.append(f"{value:.3f}" if not np.isnan(value) else "N/A")
            
            markdown += f"| **{row_model}** | {' | '.join(row_data)} |\n"
        
        # Footer
        markdown += f"\n*Generated on {timestamp} using {self.category} category results from the {self.promptset} promptset*\n"
        
        # Write to file if output_file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
        
        return markdown

def main():
    parser = argparse.ArgumentParser(description="Generate coding-specific rankings from head-to-head comparison results")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to include in the analysis")
    parser.add_argument("--output", default="coding_rankings.md", help="Output file path (default: coding_rankings.md)")
    parser.add_argument("--archive", default="test_archive", help="Path to test archive directory (default: test_archive)")
    
    args = parser.parse_args()
    
    analyzer = CodingRankAnalyzer(test_archive_dir=args.archive)
    results = analyzer.generate_rankings(args.models)
    analyzer.generate_markdown(results, args.output)
    
    print(f"Rankings generated and saved to {args.output}")


if __name__ == "__main__":
    main()