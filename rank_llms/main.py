#!/usr/bin/env python3
import os
import time
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

import typer
import ollama
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from anthropic import Anthropic, APIError
from pydantic import BaseModel, Field

# Set up logging
def setup_logging(level=logging.INFO):
    """Set up logging to file with the specified level."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rank_llms_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Create logger
    logger = logging.getLogger("rank_llms")
    logger.info(f"Logging initialized. Writing to {log_file}")
    
    return logger

# Initialize logger
logger = setup_logging()

from rank_llms.prompts import get_prompt_categories, get_prompts_from_categories, load_promptset
from rank_llms.compare import (
    ModelComparison, CategoryResult, ComparisonResult, 
    evaluate_comparison, save_comparison_result, load_comparison_result
)
from rank_llms.leaderboard import (
    generate_elo_ratings, save_leaderboard_markdown, 
    save_leaderboard_json, display_leaderboard
)
from rank_llms.analyzer import suggest_additional_tests
from rank_llms.direct_comparison import (
    DirectComparisonRanking,
    format_comparison_results_markdown,
    format_missing_comparisons_markdown
)
from rank_llms.focus_rank import (
    FocusRanking,
    format_focus_ranking_markdown
)

console = Console()
app = typer.Typer()

def query_model(model: str, prompt: str) -> Dict[str, Any]:
    """Query a model using Ollama."""
    start_time = time.time()
    logger.info(f"Querying Ollama model: {model}")
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.info(f"Received response from {model} in {response_time:.2f}s")
        
        return {
            "response": response.message.content,
            "response_time": response_time
        }
    except Exception as e:
        error_message = f"Error querying model {model}: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        
        return {
            "response": f"Error: {str(e)}",
            "response_time": time.time() - start_time
        }

def compare_models(
    anthropic_client: Anthropic,
    model_a: str,
    model_b: str,
    prompt: str,
    category: str
) -> ModelComparison:
    """Compare responses from two models for the same prompt."""
    # Query both models
    logger.info(f"Comparing models {model_a} vs {model_b} for prompt in category: {category}")
    
    # Query model A
    result_a = query_model(model_a, prompt)
    
    # Query model B
    result_b = query_model(model_b, prompt)
    
    # Evaluate the comparison
    evaluation = evaluate_comparison(
        anthropic_client,
        prompt,
        result_a["response"],
        result_b["response"],
        model_a,
        model_b,
        category
    )
    
    # Create and return the comparison
    return ModelComparison(
        model_a=model_a,
        model_b=model_b,
        prompt=prompt,
        category=category,
        response_a=result_a["response"],
        response_b=result_b["response"],
        response_time_a=result_a["response_time"],
        response_time_b=result_b["response_time"],
        winner=evaluation.get("winner"),
        reason=evaluation.get("reason")
    )

def save_markdown_report(result: ComparisonResult, output_path: str = "results.md") -> None:
    """Save detailed comparison results to a markdown file."""
    path = Path(output_path)
    
    logger.info(f"Saving detailed comparison results to {path}")
    try:
        with open(path, "w") as f:
            f.write(f"# LLM Model Comparison: {result.model_a} vs {result.model_b}\n\n")
            f.write(f"Generated on: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary table
            f.write("## Summary\n\n")
            f.write("| Category | Win % for " + result.model_a + " | Win % for " + result.model_b + " | Ties % |\n")
            f.write("|----------|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 10 + "|\n")
            
            # Add overall results
            f.write("| **Overall** | " + 
                  f"{result.overall_win_percentage_a:.1f}% | " +
                  f"{result.overall_win_percentage_b:.1f}% | " +
                  f"{100 - result.overall_win_percentage_a - result.overall_win_percentage_b:.1f}% |\n")
            
            # Add category results
            for category, cat_result in result.category_results.items():
                if cat_result.total > 0:
                    f.write(f"| {category} | " +
                          f"{cat_result.win_percentage_a:.1f}% | " +
                          f"{cat_result.win_percentage_b:.1f}% | " +
                          f"{cat_result.tie_percentage:.1f}% |\n")
            
            # Write detailed comparison results
            f.write("\n## Detailed Comparisons\n\n")
            
            # Group comparisons by category
            by_category = {}
            for comparison in result.comparisons:
                if comparison.category not in by_category:
                    by_category[comparison.category] = []
                by_category[comparison.category].append(comparison)
            
            for category, comparisons in by_category.items():
                f.write(f"### {category}\n\n")
                
                for i, comparison in enumerate(comparisons, 1):
                    f.write(f"**Prompt {i}**: {comparison.prompt}\n\n")
                    
                    # Model A response
                    f.write(f"*{comparison.model_a} (Time: {comparison.response_time_a:.2f}s)*\n\n")
                    f.write(f"```\n{comparison.response_a}\n```\n\n")
                    
                    # Model B response
                    f.write(f"*{comparison.model_b} (Time: {comparison.response_time_b:.2f}s)*\n\n")
                    f.write(f"```\n{comparison.response_b}\n```\n\n")
                    
                    # Winner
                    winner_text = ""
                    if comparison.winner == "a":
                        winner_text = f"**Winner: {comparison.model_a}**"
                    elif comparison.winner == "b":
                        winner_text = f"**Winner: {comparison.model_b}**"
                    else:
                        winner_text = "**Tie**"
                    
                    f.write(f"{winner_text}\n\n")
                    f.write(f"**Reason**: {comparison.reason}\n\n")
                    
                    f.write("---\n\n")
        
        logger.info(f"Successfully saved detailed results to {path.resolve()}")
        console.print(f"[green]Detailed results saved to {path.resolve()}")
    except Exception as e:
        error_message = f"Error saving results to {path}: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")

@app.command()
def compare(
    models: List[str] = typer.Argument(..., help="Models to compare (must be available in Ollama)"),
    num_prompts: int = typer.Option(5, help="Number of prompts to test per category"),
    promptset: str = typer.Option("basic1", help="Name of the promptset to use"),
    output: str = typer.Option("results.md", help="Output file for detailed results"),
    force_retest: bool = typer.Option(False, help="Force retesting of models even if archived results exist"),
    update_leaderboard: bool = typer.Option(True, help="Update the leaderboard with the new results"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Compare two LLM models head-to-head using Ollama and Claude for evaluation."""
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    # Ensure we have exactly two models
    if len(models) != 2:
        console.print("[red]Error: You must specify exactly two models to compare")
        logger.error("Invalid number of models provided for comparison")
        raise typer.Exit(1)
    
    model_a, model_b = models
    logger.info(f"Starting comparison of {model_a} vs {model_b}, num_prompts: {num_prompts}")
    
    # Check if ANTHROPIC_API_KEY is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        error_message = "ANTHROPIC_API_KEY environment variable not set"
        logger.error(error_message)
        console.print(f"[red]Error: {error_message}")
        raise typer.Exit(1)
    
    logger.info("Initializing Anthropic client")
    anthropic_client = Anthropic(api_key=api_key)
    
    # Check if models are available in Ollama
    logger.info("Checking available Ollama models")
    try:
        available_models = {getattr(model, "model") for model in ollama.list().models}
        logger.info(f"Found {len(available_models)} available Ollama models")
    except Exception as e:
        error_message = f"Error listing Ollama models: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        raise typer.Exit(1)
    
    for model in models:
        if model not in available_models:
            warning_message = f"Model '{model}' not found in Ollama. Available models: {', '.join(available_models)}"
            logger.warning(warning_message)
            console.print(f"[yellow]Warning: {warning_message}")
            if not typer.confirm("Continue anyway?"):
                logger.info("User chose not to continue with unavailable model")
                raise typer.Exit(1)
    
    # Check if a comparison already exists and we're not forcing a retest
    if not force_retest:
        existing_result = load_comparison_result(model_a, model_b, promptset)
        if existing_result:
            logger.info(f"Found existing comparison between {model_a} and {model_b} using promptset '{promptset}'")
            console.print(f"[green]Found existing comparison between {model_a} and {model_b} using promptset '{promptset}'")
            
            # Save the report and update the leaderboard
            save_markdown_report(existing_result, output)
            
            # Display summary of existing comparison
            console.print("\n[bold green]Existing Comparison Summary:[/bold green]")
            console.print(f"[cyan]{model_a}[/cyan] vs [cyan]{model_b}[/cyan]")
            
            # Overall results
            total_comparisons = existing_result.overall_total
            console.print(f"\nOverall: [cyan]{existing_result.model_a}[/cyan] wins {existing_result.overall_win_percentage_a:.1f}% ({existing_result.overall_wins_a}/{total_comparisons}), " + 
                         f"[cyan]{existing_result.model_b}[/cyan] wins {existing_result.overall_win_percentage_b:.1f}% ({existing_result.overall_wins_b}/{total_comparisons}), " +
                         f"Ties {100 - existing_result.overall_win_percentage_a - existing_result.overall_win_percentage_b:.1f}% ({existing_result.overall_ties}/{total_comparisons})")
            
            # Category results
            for category, cat_result in existing_result.category_results.items():
                if cat_result.total > 0:
                    console.print(f"{category}: [cyan]{model_a}[/cyan] wins {cat_result.win_percentage_a:.1f}% ({cat_result.wins_a}/{cat_result.total}), " + 
                                 f"[cyan]{model_b}[/cyan] wins {cat_result.win_percentage_b:.1f}% ({cat_result.wins_b}/{cat_result.total}), " +
                                 f"Ties {cat_result.tie_percentage:.1f}% ({cat_result.ties}/{cat_result.total})")
            
            # Provide a link to the full report
            console.print(f"\n[bold]Full comparison report saved to: [link={output}]{output}[/link][/bold]")
            
            if update_leaderboard:
                logger.info(f"Updating leaderboard with existing results for promptset '{promptset}'")
                elo_system = generate_elo_ratings(force_refresh=True, promptset=promptset)
                save_leaderboard_markdown(elo_system)
                save_leaderboard_json(elo_system)
                display_leaderboard(elo_system)
            
            return existing_result
    
    # Get prompts
    logger.info(f"Getting prompt categories and prompts from promptset '{promptset}'")
    categories = get_prompt_categories(promptset_name=promptset)
    all_prompts = get_prompts_from_categories(categories, max_per_category=num_prompts, promptset_name=promptset)
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    logger.info(f"Selected {total_prompts} prompts across {len(categories)} categories from promptset '{promptset}'")
    
    # Initialize results
    category_results = {
        category: CategoryResult(
            category=category,
            model_a=model_a,
            model_b=model_b
        )
        for category in categories
    }
    comparisons = []
    
    # Run the comparisons
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("[cyan]Comparing models...", total=total_prompts)
        
        for category, prompts in all_prompts.items():
            logger.info(f"Testing category: {category} with {len(prompts)} prompts")
            
            for prompt in prompts:
                progress.update(task, description=f"[cyan]Comparing {model_a} vs {model_b} on {category} prompt")
                logger.info(f"Comparing prompt: {prompt[:50]}...")
                
                # Compare the models
                comparison = compare_models(
                    anthropic_client,
                    model_a,
                    model_b,
                    prompt,
                    category
                )
                
                # Update category results
                if comparison.winner == "a":
                    category_results[category].wins_a += 1
                elif comparison.winner == "b":
                    category_results[category].wins_b += 1
                else:
                    category_results[category].ties += 1
                
                # Add to comparisons list
                comparisons.append(comparison)
                
                # Update progress
                progress.update(task, advance=1)
    
    # Create the result
    result = ComparisonResult(
        model_a=model_a,
        model_b=model_b,
        category_results=category_results,
        comparisons=comparisons
    )
    
    # Save the result for future reference
    save_comparison_result(result, promptset=promptset)
    
    # Save the detailed report
    save_markdown_report(result, output)
    
    # Update the leaderboard
    if update_leaderboard:
        logger.info(f"Updating leaderboard with new results for promptset '{promptset}'")
        elo_system = generate_elo_ratings(force_refresh=True, promptset=promptset)
        save_leaderboard_markdown(elo_system)
        save_leaderboard_json(elo_system)
        display_leaderboard(elo_system)
    
    # Display summary
    console.print("\n[bold green]Comparison Summary:[/bold green]")
    console.print(f"[cyan]{model_a}[/cyan] vs [cyan]{model_b}[/cyan]")
    
    # Overall results
    total_comparisons = result.overall_total
    console.print(f"\nOverall: [cyan]{result.model_a}[/cyan] wins {result.overall_win_percentage_a:.1f}% ({result.overall_wins_a}/{total_comparisons}), " + 
                 f"[cyan]{result.model_b}[/cyan] wins {result.overall_win_percentage_b:.1f}% ({result.overall_wins_b}/{total_comparisons}), " +
                 f"Ties {100 - result.overall_win_percentage_a - result.overall_win_percentage_b:.1f}% ({result.overall_ties}/{total_comparisons})")
    
    # Category results
    for category, cat_result in result.category_results.items():
        if cat_result.total > 0:
            console.print(f"{category}: [cyan]{model_a}[/cyan] wins {cat_result.win_percentage_a:.1f}% ({cat_result.wins_a}/{cat_result.total}), " + 
                         f"[cyan]{model_b}[/cyan] wins {cat_result.win_percentage_b:.1f}% ({cat_result.wins_b}/{cat_result.total}), " +
                         f"Ties {cat_result.tie_percentage:.1f}% ({cat_result.ties}/{cat_result.total})")
    
    return result

@app.command()
def promptset_info(
    promptset: str = typer.Option("basic1", help="Name of the promptset to display information about"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Display information about a promptset."""
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    try:
        # Load the promptset
        promptset_data = load_promptset(promptset)
        
        # Display information
        console.print(f"\n[bold green]Promptset: {promptset}[/bold green]")
        console.print(f"Categories: {len(promptset_data)}")
        
        for category, prompts in promptset_data.items():
            console.print(f"\n[bold cyan]{category}[/bold cyan] ({len(prompts)} prompts):")
            
            for i, prompt in enumerate(prompts, 1):
                # Truncate long prompts for display
                truncated = prompt[:100] + "..." if len(prompt) > 100 else prompt
                console.print(f"  {i}. {truncated}")
        
        return promptset_data
    
    except Exception as e:
        console.print(f"[red]Error loading promptset {promptset}: {e}")
        raise typer.Exit(1)

@app.command()
def suggest_tests(
    promptset: str = typer.Option("basic1", help="Name of the promptset to analyze"),
    max_suggestions: int = typer.Option(10, help="Maximum number of suggestions to display"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Analyze test archives and suggest additional tests to run for increasing ranking confidence."""
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    # Suggest additional tests
    logger.info(f"Analyzing test archives for promptset '{promptset}'")
    suggestions = suggest_additional_tests(promptset, max_suggestions)
    
    return suggestions

@app.command()
def leaderboard(
    promptset: str = typer.Option("basic1", help="Name of the promptset to use for the leaderboard"),
    output: str = typer.Option(None, help="Output file for leaderboard (defaults to leaderboard/<promptset>_leaderboard.md)"),
    json_output: str = typer.Option(None, help="JSON output file for leaderboard (defaults to leaderboard/<promptset>_leaderboard.json)"),
    force_refresh: bool = typer.Option(False, help="Force refresh of ELO ratings from all comparison results"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Generate and display the leaderboard based on all comparison results."""
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    logger.info(f"Generating leaderboard for promptset '{promptset}'")
    
    # Generate ELO ratings
    elo_system = generate_elo_ratings(force_refresh=force_refresh, promptset=promptset)
    
    # Save leaderboard files
    save_leaderboard_markdown(elo_system, output)
    save_leaderboard_json(elo_system, json_output)
    
    # Display leaderboard
    display_leaderboard(elo_system)
    
    return elo_system

@app.command()
def ranksubset(
    models: List[str] = typer.Argument(..., help="List of models to rank using direct comparison results"),
    promptset: str = typer.Option("basic1", help="Name of the promptset to use for the analysis"),
    output: str = typer.Option(None, help="Output file for the markdown table (defaults to stdout)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """
    Rank a subset of models using direct head-to-head comparison results.
    All models in the list must have been compared with each other.
    The output shows the actual win probabilities based on existing test results.
    """
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    # Check if we have enough models
    if len(models) < 2:
        console.print("[red]Error: You must specify at least two models to rank")
        logger.error("Not enough models provided for subset ranking")
        raise typer.Exit(1)
    
    logger.info(f"Generating direct comparison rankings for {len(models)} models using promptset '{promptset}'")
    
    try:
        # Generate direct comparison rankings
        ranking = DirectComparisonRanking()
        all_comparisons_available = ranking.compute_rankings(models, promptset)
        
        if not all_comparisons_available:
            # Generate missing comparisons instructions
            markdown = format_missing_comparisons_markdown(ranking, promptset)
            console.print("[yellow]Warning: Not all models have been directly compared with each other.")
            console.print("The following comparisons are needed for a complete analysis:")
            
            # Format commands for missing comparisons
            commands = ranking.get_missing_comparison_commands(promptset)
            for i, command in enumerate(commands, 1):
                console.print(f"[cyan]{i}. {command}")
            
            # Save missing comparisons to file if requested
            if output:
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, "w") as f:
                    f.write(markdown)
                
                logger.info(f"Saved missing comparisons list to {path}")
                console.print(f"[green]Missing comparisons list saved to {path}")
                
            return ranking
        
        # Format results as markdown
        markdown = format_comparison_results_markdown(ranking)
        
        # Output results
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                f.write(markdown)
            
            logger.info(f"Saved direct comparison rankings to {path}")
            console.print(f"[green]Direct comparison rankings saved to {path}")
        else:
            # Print to console
            console.print(markdown)
    
    except Exception as e:
        error_message = f"Error generating direct comparison rankings: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        raise typer.Exit(1)
    
    return ranking
    
@app.command()
def focusrank(
    focus_model: str = typer.Argument(..., help="The model to use as the focus for ranking"),
    promptset: str = typer.Option("basic1", help="Name of the promptset to use for the analysis"),
    depth: int = typer.Option(3, help="Maximum depth for transitive relationships (1 = direct only)"),
    output: str = typer.Option(None, help="Output file for the markdown table (defaults to stdout)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """
    Rank models based on their performance against a specific focus model.
    
    This command uses win ratio as the primary metric: other_model_wins / focus_model_wins.
    A ratio > 1.0 means the model outperforms the focus model, while < 1.0 means it underperforms.
    
    If no direct comparison exists, transitive relationships can be used to estimate rankings.
    """
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    logger.info(f"Generating focus-based rankings with focus model '{focus_model}' using promptset '{promptset}'")
    
    try:
        # Create focus ranking
        ranking = FocusRanking(focus_model)
        win_ratios = ranking.compute_rankings(promptset=promptset, max_depth=depth)
        
        if not win_ratios:
            console.print(f"[red]Error: Focus model '{focus_model}' not found in any comparisons for promptset '{promptset}'")
            raise typer.Exit(1)
        
        # Generate ranking table
        ranking_table = ranking.get_ranking_table(win_ratios)
        
        # Get raw comparison data
        comparison_data = ranking.get_raw_comparison_data()
        
        # Format results as markdown
        markdown = format_focus_ranking_markdown(
            focus_model, 
            ranking_table, 
            comparison_data,
            promptset=promptset,
            max_depth=depth
        )
        
        # Output results
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                f.write(markdown)
            
            logger.info(f"Saved focus-based rankings to {path}")
            console.print(f"[green]Focus-based rankings saved to {path}")
        else:
            # Print to console
            console.print(markdown)
    
    except Exception as e:
        error_message = f"Error generating focus-based rankings: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        raise typer.Exit(1)
    
    return ranking_table

if __name__ == "__main__":
    app()