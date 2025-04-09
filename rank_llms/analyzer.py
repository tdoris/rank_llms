"""
Analyzer module for analyzing test archives and suggesting additional tests.
"""

import logging
import itertools
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import json
import math
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from rank_llms.prompts import get_prompt_categories, load_promptset
from rank_llms.leaderboard import find_all_comparison_files
from rank_llms.compare import load_comparison_result
from rank_llms.elo import EloRatingSystem

# Get logger
logger = logging.getLogger("rank_llms")
console = Console()

class ConfidenceAnalyzer:
    """
    Analyzes test archives and suggests additional tests to run
    to increase confidence in model rankings.
    """
    
    def __init__(self, promptset: str = "basic1"):
        """
        Initialize the analyzer with a specific promptset.
        
        Args:
            promptset: Name of the promptset to analyze
        """
        self.promptset = promptset
        self.comparison_files = find_all_comparison_files(promptset)
        self.elo_file = f"leaderboard/{promptset}_elo_ratings.json"
        self.models = set()
        self.comparisons = {}  # Dict mapping model pairs to comparison counts
        self.category_comparisons = defaultdict(lambda: defaultdict(int))  # Dict mapping categories to model pairs to counts
        self.model_category_counts = defaultdict(lambda: defaultdict(int))  # Dict mapping models to categories to counts
        
        # Load ELO ratings if they exist
        self.elo_ratings = None
        if Path(self.elo_file).exists():
            self.elo_ratings = EloRatingSystem.load_ratings(self.elo_file, promptset)
        
        # Load all models and comparison data
        self._load_comparison_data()
    
    def _load_comparison_data(self):
        """Load all comparison data from the archive."""
        logger.info(f"Loading comparison data for promptset '{self.promptset}'")
        
        for file_path in self.comparison_files:
            try:
                # Extract model names from filename
                filename = file_path.name
                model_part = filename.replace(".json", "")
                models = model_part.split("__vs__")
                
                if len(models) != 2:
                    logger.warning(f"Invalid comparison filename format: {filename}")
                    continue
                
                model_a = models[0].replace("_", ":")
                model_b = models[1].replace("_", ":")
                
                # Add models to the set
                self.models.add(model_a)
                self.models.add(model_b)
                
                # Load the comparison result
                result = load_comparison_result(model_a, model_b, self.promptset)
                if result:
                    # Sort model names for consistent tracking
                    model_pair = tuple(sorted([model_a, model_b]))
                    
                    # Track total comparisons for this pair
                    self.comparisons[model_pair] = self.comparisons.get(model_pair, 0) + len(result.comparisons)
                    
                    # Track comparisons by category
                    for category, cat_result in result.category_results.items():
                        if cat_result.total > 0:
                            self.category_comparisons[category][model_pair] += cat_result.total
                            self.model_category_counts[model_a][category] += cat_result.total
                            self.model_category_counts[model_b][category] += cat_result.total
            
            except Exception as e:
                logger.error(f"Error processing comparison file {file_path}: {e}")
        
        logger.info(f"Loaded data for {len(self.models)} models and {len(self.comparisons)} model pairs")
    
    def get_missing_comparisons(self) -> List[Tuple[str, str]]:
        """
        Identify pairs of models that have not been compared yet.
        
        Returns:
            List of model pairs (as tuples) that need to be compared
        """
        all_possible_pairs = list(itertools.combinations(sorted(self.models), 2))
        existing_pairs = set(self.comparisons.keys())
        missing_pairs = [pair for pair in all_possible_pairs if pair not in existing_pairs]
        
        return missing_pairs
    
    def get_low_confidence_pairs(self, min_comparisons: int = 5) -> List[Tuple[str, str, int]]:
        """
        Identify pairs of models with too few comparisons for high confidence.
        
        Args:
            min_comparisons: Minimum number of comparisons needed for confidence
            
        Returns:
            List of tuples containing (model_a, model_b, current_count)
        """
        low_confidence = []
        
        for pair, count in self.comparisons.items():
            if count < min_comparisons:
                low_confidence.append((pair[0], pair[1], count))
        
        # Sort by count (ascending) so pairs with fewest comparisons come first
        low_confidence.sort(key=lambda x: x[2])
        
        return low_confidence
    
    def get_close_rating_pairs(self, max_diff: int = 50) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of models with close ELO ratings that need more comparisons.
        
        Args:
            max_diff: Maximum ELO difference to consider models as having close ratings
            
        Returns:
            List of tuples containing (model_a, model_b, rating_diff)
        """
        if not self.elo_ratings:
            logger.warning("No ELO ratings available for this promptset")
            return []
        
        close_pairs = []
        
        # Get model ratings
        for model_a, model_b in itertools.combinations(sorted(self.models), 2):
            rating_a = self.elo_ratings.get_rating(model_a)
            rating_b = self.elo_ratings.get_rating(model_b)
            
            diff = abs(rating_a - rating_b)
            
            if diff <= max_diff:
                close_pairs.append((model_a, model_b, diff))
        
        # Sort by rating difference (ascending)
        close_pairs.sort(key=lambda x: x[2])
        
        return close_pairs
    
    def get_category_gaps(self, min_per_category: int = 3) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Identify model pairs that need more comparisons in specific categories.
        
        Args:
            min_per_category: Minimum number of comparisons needed per category
            
        Returns:
            Dict mapping categories to lists of (model_a, model_b, current_count) tuples
        """
        categories = get_prompt_categories(self.promptset)
        gaps = {category: [] for category in categories}
        
        for category in categories:
            for pair in self.comparisons.keys():
                count = self.category_comparisons[category].get(pair, 0)
                if count < min_per_category:
                    gaps[category].append((pair[0], pair[1], count))
        
        # Sort each category list by count (ascending)
        for category in gaps:
            gaps[category].sort(key=lambda x: x[2])
        
        return gaps
    
    def get_underrepresented_models(self) -> List[Tuple[str, int]]:
        """
        Identify models that are underrepresented in the comparisons.
        
        Returns:
            List of tuples containing (model_name, comparison_count)
        """
        model_counts = defaultdict(int)
        
        # Count total comparisons for each model
        for model_a, model_b in self.comparisons.keys():
            count = self.comparisons[(model_a, model_b)]
            model_counts[model_a] += count
            model_counts[model_b] += count
        
        # Sort by comparison count (ascending)
        models_by_count = [(model, count) for model, count in model_counts.items()]
        models_by_count.sort(key=lambda x: x[1])
        
        return models_by_count
    
    def generate_suggestions(self, 
                            min_comparisons: int = 5, 
                            min_per_category: int = 2,
                            max_rating_diff: int = 50,
                            max_suggestions: int = 10) -> List[Dict]:
        """
        Generate a prioritized list of suggested comparisons to run.
        
        Args:
            min_comparisons: Minimum comparisons needed per model pair
            min_per_category: Minimum comparisons needed per category
            max_rating_diff: Maximum ELO difference for close-rating pairs
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries with models, reason, and priority
        """
        suggestions = []
        
        # 1. Missing comparisons (highest priority)
        missing = self.get_missing_comparisons()
        for model_a, model_b in missing[:max_suggestions]:
            suggestions.append({
                "model_a": model_a,
                "model_b": model_b,
                "reason": "These models have never been compared",
                "priority": 1,
                "promptset": self.promptset
            })
        
        # 2. Low confidence pairs
        low_conf = self.get_low_confidence_pairs(min_comparisons)
        for model_a, model_b, count in low_conf[:max_suggestions]:
            if (model_a, model_b) not in [(s["model_a"], s["model_b"]) for s in suggestions] and \
               (model_b, model_a) not in [(s["model_a"], s["model_b"]) for s in suggestions]:
                suggestions.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "reason": f"Only {count} comparisons (recommended: {min_comparisons})",
                    "priority": 2,
                    "promptset": self.promptset
                })
        
        # 3. Close rating pairs
        if self.elo_ratings:
            close_pairs = self.get_close_rating_pairs(max_rating_diff)
            for model_a, model_b, diff in close_pairs[:max_suggestions]:
                if (model_a, model_b) not in [(s["model_a"], s["model_b"]) for s in suggestions] and \
                   (model_b, model_a) not in [(s["model_a"], s["model_b"]) for s in suggestions]:
                    suggestions.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "reason": f"Close ELO ratings (diff: {diff:.1f})",
                        "priority": 3,
                        "promptset": self.promptset
                    })
        
        # 4. Category gaps
        category_gaps = self.get_category_gaps(min_per_category)
        for category, pairs in category_gaps.items():
            for model_a, model_b, count in pairs[:max_suggestions//len(category_gaps)]:
                if (model_a, model_b) not in [(s["model_a"], s["model_b"]) for s in suggestions] and \
                   (model_b, model_a) not in [(s["model_a"], s["model_b"]) for s in suggestions]:
                    suggestions.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "reason": f"Only {count} comparisons in '{category}' category",
                        "priority": 4,
                        "category": category,
                        "promptset": self.promptset
                    })
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x["priority"])
        
        return suggestions[:max_suggestions]
    
    def display_suggestions(self, max_suggestions: int = 10):
        """
        Display suggested comparisons in a formatted table.
        
        Args:
            max_suggestions: Maximum number of suggestions to display
        """
        suggestions = self.generate_suggestions(max_suggestions=max_suggestions)
        
        console.print(f"\n[bold green]Suggested Additional Tests for Promptset '{self.promptset}'[/bold green]")
        
        if not suggestions:
            console.print("No additional tests needed at this time.")
            return
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Priority", style="dim", width=6)
        table.add_column("Model A", style="cyan")
        table.add_column("Model B", style="cyan")
        table.add_column("Reason", style="green")
        table.add_column("Category", style="yellow")
        
        # Add rows
        for i, suggestion in enumerate(suggestions, 1):
            table.add_row(
                str(i),
                suggestion["model_a"],
                suggestion["model_b"],
                suggestion["reason"],
                suggestion.get("category", "All")
            )
        
        console.print(table)
        
        # Show command suggestions
        console.print("\n[bold]Example commands to run:[/bold]")
        for i, suggestion in enumerate(suggestions[:3], 1):
            model_a = suggestion["model_a"]
            model_b = suggestion["model_b"]
            promptset = suggestion["promptset"]
            category_opt = f" --categories {suggestion['category']}" if "category" in suggestion else ""
            
            console.print(f"[dim]{i}.[/dim] [bold]rank-llms compare {model_a} {model_b} --promptset {promptset}{category_opt}[/bold]")
    
    def get_model_summary(self) -> Dict:
        """
        Get a summary of model comparisons.
        
        Returns:
            Dictionary with model summary information
        """
        # Count comparisons per model
        model_comparison_counts = defaultdict(int)
        for (model_a, model_b), count in self.comparisons.items():
            model_comparison_counts[model_a] += count
            model_comparison_counts[model_b] += count
        
        # Get category distribution per model
        model_category_distribution = {}
        for model, categories in self.model_category_counts.items():
            total = sum(categories.values())
            if total > 0:
                distribution = {cat: (count / total) * 100 for cat, count in categories.items()}
                model_category_distribution[model] = distribution
        
        # Get model ratings if available
        model_ratings = {}
        if self.elo_ratings:
            for model in self.models:
                model_ratings[model] = self.elo_ratings.get_rating(model)
        
        return {
            "total_models": len(self.models),
            "total_comparisons": sum(self.comparisons.values()),
            "model_comparison_counts": dict(model_comparison_counts),
            "model_category_distribution": model_category_distribution,
            "model_ratings": model_ratings
        }
    
    def display_model_summary(self):
        """Display a summary of model comparisons."""
        summary = self.get_model_summary()
        
        console.print(f"\n[bold green]Model Comparison Summary for Promptset '{self.promptset}'[/bold green]")
        console.print(f"Total Models: {summary['total_models']}")
        console.print(f"Total Comparisons: {summary['total_comparisons']}")
        
        # Display model comparison counts
        console.print("\n[bold cyan]Comparisons Per Model:[/bold cyan]")
        model_counts = [(model, count) for model, count in summary["model_comparison_counts"].items()]
        model_counts.sort(key=lambda x: x[1], reverse=True)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Model", style="cyan")
        table.add_column("Comparisons", justify="right")
        table.add_column("ELO Rating", justify="right")
        
        for i, (model, count) in enumerate(model_counts, 1):
            rating = summary["model_ratings"].get(model, "N/A")
            rating_str = f"{rating:.0f}" if isinstance(rating, (int, float)) else rating
            table.add_row(str(i), model, str(count), rating_str)
        
        console.print(table)
        
        # Display category distribution for top models
        console.print("\n[bold cyan]Category Distribution for Top Models:[/bold cyan]")
        top_models = [model for model, _ in model_counts[:5]]
        
        categ_table = Table(show_header=True, header_style="bold magenta")
        categ_table.add_column("Model", style="cyan")
        
        # Get all categories
        all_categories = set()
        for model in top_models:
            if model in summary["model_category_distribution"]:
                all_categories.update(summary["model_category_distribution"][model].keys())
        
        # Add category columns
        for category in sorted(all_categories):
            categ_table.add_column(category, justify="right")
        
        # Add rows for each model
        for model in top_models:
            if model in summary["model_category_distribution"]:
                row = [model]
                for category in sorted(all_categories):
                    pct = summary["model_category_distribution"][model].get(category, 0)
                    row.append(f"{pct:.1f}%")
                categ_table.add_row(*row)
        
        console.print(categ_table)


def suggest_additional_tests(promptset: str = "basic1", max_suggestions: int = 10):
    """
    Analyze test archives and suggest additional tests to run.
    
    Args:
        promptset: Name of the promptset to analyze
        max_suggestions: Maximum number of suggestions to display
    """
    analyzer = ConfidenceAnalyzer(promptset)
    analyzer.display_model_summary()
    analyzer.display_suggestions(max_suggestions)
    
    return analyzer.generate_suggestions(max_suggestions=max_suggestions)