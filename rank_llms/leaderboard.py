"""Leaderboard generation for LLM model rankings."""

import json
import logging
from typing import Dict, List, Any, Set
from pathlib import Path
import datetime
import glob

import pandas as pd
from rich.console import Console
from rich.table import Table

from rank_llms.elo import EloRatingSystem
from rank_llms.compare import load_comparison_result, update_elo_ratings

# Get logger
logger = logging.getLogger("rank_llms")
console = Console()

def find_all_comparison_files(promptset: str = "basic1") -> List[Path]:
    """Find all comparison result files in the archive for a specific promptset."""
    pattern = str(Path("test_archive") / promptset / "comparisons" / "*.json")
    return [Path(file) for file in glob.glob(pattern)]

def generate_elo_ratings(force_refresh: bool = False, promptset: str = "basic1") -> EloRatingSystem:
    """Generate ELO ratings from all comparison results for a specific promptset."""
    elo_file = f"leaderboard/{promptset}_elo_ratings.json"
    Path("leaderboard").mkdir(parents=True, exist_ok=True)
    
    # If not forcing refresh and ratings exist, just load them
    if not force_refresh and Path(elo_file).exists():
        return EloRatingSystem.load_ratings(elo_file, promptset=promptset)
    
    # Find all comparison files for this promptset
    comparison_files = find_all_comparison_files(promptset)
    logger.info(f"Found {len(comparison_files)} comparison files for promptset '{promptset}'")
    
    # Load all comparison results
    results = []
    for file_path in comparison_files:
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
            
            # Load the comparison result
            result = load_comparison_result(model_a, model_b, promptset=promptset)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing comparison file {file_path}: {e}")
    
    # Update ELO ratings based on all comparisons
    elo_system = update_elo_ratings(results, elo_file)
    logger.info(f"Updated ELO ratings based on {len(results)} comparison results for promptset '{promptset}'")
    
    return elo_system

def save_leaderboard_markdown(elo_system: EloRatingSystem, output_path: str = None) -> None:
    """Save the leaderboard as a markdown file."""
    # Use promptset-specific output path if not provided
    if output_path is None:
        output_path = f"leaderboard/{elo_system.promptset}_leaderboard.md"
    # Get all models and separate regular models from category-specific ones
    all_models = elo_system.get_all_models()
    regular_models = {model for model in all_models if "__" not in model}
    categories = set()
    
    for model in all_models:
        if "__" in model:
            _, category = model.split("__", 1)
            categories.add(category)
    
    # Create the markdown content
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# LLM Model Leaderboard\n\n"
    content += f"Generated on: {now}\n\n"
    
    # Overall rankings
    content += "## Overall Rankings\n\n"
    content += "| Rank | Model | ELO Rating |\n"
    content += "|------|-------|------------|\n"
    
    # Get and sort overall rankings
    overall_rankings = [
        (model, elo_system.get_rating(model))
        for model in regular_models
    ]
    overall_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, rating) in enumerate(overall_rankings, 1):
        content += f"| {i} | {model} | {rating:.0f} |\n"
    
    # Category rankings
    for category in sorted(categories):
        content += f"\n## {category} Rankings\n\n"
        content += "| Rank | Model | ELO Rating |\n"
        content += "|------|-------|------------|\n"
        
        # Get and sort category rankings
        category_rankings = [
            (model, elo_system.get_rating(f"{model}__{category}"))
            for model in regular_models
            if f"{model}__{category}" in all_models
        ]
        category_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, rating) in enumerate(category_rankings, 1):
            content += f"| {i} | {model} | {rating:.0f} |\n"
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(content)
    
    logger.info(f"Saved leaderboard to {output_file}")

def save_leaderboard_json(elo_system: EloRatingSystem, output_path: str = None) -> None:
    """Save the leaderboard as a JSON file."""
    # Use promptset-specific output path if not provided
    if output_path is None:
        output_path = f"leaderboard/{elo_system.promptset}_leaderboard.json"
    # Get all models and separate regular models from category-specific ones
    all_models = elo_system.get_all_models()
    regular_models = {model for model in all_models if "__" not in model}
    categories = set()
    
    for model in all_models:
        if "__" in model:
            _, category = model.split("__", 1)
            categories.add(category)
    
    # Create the JSON structure
    leaderboard = {
        "generated_at": datetime.datetime.now().isoformat(),
        "overall": [],
        "categories": {}
    }
    
    # Overall rankings
    overall_rankings = [
        {"model": model, "rating": elo_system.get_rating(model)}
        for model in regular_models
    ]
    overall_rankings.sort(key=lambda x: x["rating"], reverse=True)
    leaderboard["overall"] = overall_rankings
    
    # Category rankings
    for category in categories:
        category_rankings = [
            {"model": model, "rating": elo_system.get_rating(f"{model}__{category}")}
            for model in regular_models
            if f"{model}__{category}" in all_models
        ]
        category_rankings.sort(key=lambda x: x["rating"], reverse=True)
        leaderboard["categories"][category] = category_rankings
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(leaderboard, f, indent=2)
    
    logger.info(f"Saved leaderboard JSON to {output_file}")

def display_leaderboard(elo_system: EloRatingSystem) -> None:
    """Display the leaderboard in the console."""
    # Get all models and separate regular models from category-specific ones
    all_models = elo_system.get_all_models()
    regular_models = {model for model in all_models if "__" not in model}
    categories = set()
    
    for model in all_models:
        if "__" in model:
            _, category = model.split("__", 1)
            categories.add(category)
    
    # Display overall rankings
    console.print("\n[bold green]Overall Rankings:[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Model", style="cyan")
    table.add_column("ELO Rating", justify="right")
    
    # Get and sort overall rankings
    overall_rankings = [
        (model, elo_system.get_rating(model))
        for model in regular_models
    ]
    overall_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, rating) in enumerate(overall_rankings, 1):
        table.add_row(str(i), model, f"{rating:.0f}")
    
    console.print(table)
    
    # Display category rankings
    for category in sorted(categories):
        console.print(f"\n[bold green]{category} Rankings:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Model", style="cyan")
        table.add_column("ELO Rating", justify="right")
        
        # Get and sort category rankings
        category_rankings = [
            (model, elo_system.get_rating(f"{model}__{category}"))
            for model in regular_models
            if f"{model}__{category}" in all_models
        ]
        category_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, rating) in enumerate(category_rankings, 1):
            table.add_row(str(i), model, f"{rating:.0f}")
        
        console.print(table)