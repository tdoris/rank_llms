"""Comparative evaluation of LLM model responses."""

import os
import json
import time
import pickle
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging

from anthropic import Anthropic, APIError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from pydantic import BaseModel, Field

from rank_llms.prompts import get_prompt_categories, get_prompts_from_categories
import rank_llms.elo as elo

# Get logger
logger = logging.getLogger("rank_llms")
console = Console()

class ModelComparison(BaseModel):
    """A comparison between two model responses for the same prompt."""
    model_a: str
    model_b: str
    prompt: str
    category: str
    response_a: str
    response_b: str
    response_time_a: float
    response_time_b: float
    winner: Optional[str] = None  # 'a', 'b', or None (tie)
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class CategoryResult(BaseModel):
    """Results for a single category comparison between two models."""
    category: str
    model_a: str
    model_b: str
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    
    @property
    def total(self) -> int:
        """Total number of comparisons."""
        return self.wins_a + self.wins_b + self.ties
    
    @property
    def win_percentage_a(self) -> float:
        """Percentage of comparisons won by model A."""
        if self.total == 0:
            return 0
        return (self.wins_a / self.total) * 100
    
    @property
    def win_percentage_b(self) -> float:
        """Percentage of comparisons won by model B."""
        if self.total == 0:
            return 0
        return (self.wins_b / self.total) * 100
    
    @property
    def tie_percentage(self) -> float:
        """Percentage of comparisons that ended in a tie."""
        if self.total == 0:
            return 0
        return (self.ties / self.total) * 100

class ComparisonResult(BaseModel):
    """Results of a comparison between two models across multiple categories."""
    model_a: str
    model_b: str
    category_results: Dict[str, CategoryResult]
    comparisons: List[ModelComparison]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @property
    def overall_wins_a(self) -> int:
        """Total number of wins for model A across all categories."""
        return sum(result.wins_a for result in self.category_results.values())
    
    @property
    def overall_wins_b(self) -> int:
        """Total number of wins for model B across all categories."""
        return sum(result.wins_b for result in self.category_results.values())
    
    @property
    def overall_ties(self) -> int:
        """Total number of ties across all categories."""
        return sum(result.ties for result in self.category_results.values())
    
    @property
    def overall_total(self) -> int:
        """Total number of comparisons across all categories."""
        return self.overall_wins_a + self.overall_wins_b + self.overall_ties
    
    @property
    def overall_win_percentage_a(self) -> float:
        """Overall percentage of comparisons won by model A."""
        if self.overall_total == 0:
            return 0
        return (self.overall_wins_a / self.overall_total) * 100
    
    @property
    def overall_win_percentage_b(self) -> float:
        """Overall percentage of comparisons won by model B."""
        if self.overall_total == 0:
            return 0
        return (self.overall_wins_b / self.overall_total) * 100

def get_comparison_file_path(model_a: str, model_b: str) -> Path:
    """Get the path to the comparison result file for two models."""
    # Sort model names to ensure consistent file naming regardless of order
    models = sorted([model_a, model_b])
    filename = f"{models[0].replace(':', '_').replace('/', '_')}__vs__{models[1].replace(':', '_').replace('/', '_')}.pkl"
    return Path("test_archive") / "comparisons" / filename

def save_comparison_result(result: ComparisonResult) -> None:
    """Save a comparison result to a file."""
    file_path = get_comparison_file_path(result.model_a, result.model_b)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Saved comparison result to {file_path}")
    except Exception as e:
        logger.error(f"Error saving comparison result: {e}")

def load_comparison_result(model_a: str, model_b: str) -> Optional[ComparisonResult]:
    """Load a comparison result from a file if it exists."""
    file_path = get_comparison_file_path(model_a, model_b)
    
    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                result = pickle.load(f)
            logger.info(f"Loaded comparison result from {file_path}")
            return result
        except Exception as e:
            logger.error(f"Error loading comparison result: {e}")
    
    return None

def evaluate_comparison(
    anthropic_client: Anthropic, 
    prompt: str, 
    response_a: str, 
    response_b: str, 
    model_a: str,
    model_b: str,
    category: str
) -> Dict[str, Any]:
    """
    Evaluate two model responses against each other using Claude.
    
    Returns a dictionary with the winner ('a', 'b', or None for tie) and reason.
    """
    evaluation_prompt = f"""You are evaluating two AI assistant responses to the same user query. 
Compare the responses and decide which one is better.

Category: {category}

User Query: {prompt}

Response from Model A ({model_a}):
```
{response_a}
```

Response from Model B ({model_b}):
```
{response_b}
```

Provide your evaluation in the following JSON format:
{{
  "winner": "a" or "b" or "tie",
  "reason": "<explanation of why you chose this response as better>"
}}

Focus on accuracy, helpfulness, clarity, depth, and appropriateness. Be as objective as possible.
If both responses are of equal quality, you may declare a tie.
"""
    
    logger.info(f"Calling Anthropic API to evaluate comparison in category: {category}")
    try:
        completion = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=0,
            system="Evaluate the quality of the given responses. Return only valid JSON.",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        result = completion.content[0].text
        logger.info(f"Received comparative evaluation from Anthropic API")
        
        try:
            parsed_result = json.loads(result)
            winner = parsed_result.get("winner")
            
            if winner not in ['a', 'b', 'tie', None]:
                logger.warning(f"Invalid winner value: {winner}, defaulting to 'tie'")
                parsed_result["winner"] = "tie"
            
            logger.info(f"Evaluation result: {parsed_result.get('winner', 'unknown')}")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Anthropic API: {e}")
            logger.error(f"Raw response: {result}")
            return {"winner": "tie", "reason": f"JSON parsing error: {str(e)}"}
    except APIError as e:
        error_message = f"Anthropic API error: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        return {"winner": "tie", "reason": error_message}
    except Exception as e:
        error_message = f"Error evaluating comparison: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        return {"winner": "tie", "reason": f"Evaluation error: {str(e)}"}

def update_elo_ratings(results: List[ComparisonResult], elo_file: str = "leaderboard/elo_ratings.json") -> elo.EloRatingSystem:
    """Update ELO ratings based on comparison results."""
    # Initialize or load ELO system
    elo_system = elo.EloRatingSystem.load_ratings(elo_file)
    
    # Process each comparison result
    for result in results:
        model_a = result.model_a
        model_b = result.model_b
        
        # Update overall rating
        elo_system.register_match_result(
            model_a=model_a,
            model_b=model_b,
            wins_a=result.overall_wins_a,
            wins_b=result.overall_wins_b,
            draws=result.overall_ties
        )
        
        # Update category-specific ratings
        for category, category_result in result.category_results.items():
            if category_result.total > 0:
                # Create category-specific model names
                cat_model_a = f"{model_a}__{category}"
                cat_model_b = f"{model_b}__{category}"
                
                # Update category-specific ratings
                elo_system.register_match_result(
                    model_a=cat_model_a,
                    model_b=cat_model_b,
                    wins_a=category_result.wins_a,
                    wins_b=category_result.wins_b,
                    draws=category_result.ties,
                    category=category
                )
    
    # Save updated ratings
    Path(elo_file).parent.mkdir(parents=True, exist_ok=True)
    elo_system.save_ratings(elo_file)
    
    return elo_system

def generate_leaderboard(elo_system: elo.EloRatingSystem) -> Dict[str, Any]:
    """Generate a leaderboard based on ELO ratings."""
    # Get all models
    all_models = elo_system.get_all_models()
    
    # Separate regular models from category-specific models
    regular_models = set()
    category_models = {}
    
    for model in all_models:
        if "__" in model:
            base_model, category = model.split("__", 1)
            if category not in category_models:
                category_models[category] = set()
            category_models[category].add(base_model)
            regular_models.add(base_model)
        else:
            regular_models.add(model)
    
    # Generate overall rankings
    overall_rankings = [
        {"model": model, "rating": rating}
        for model, rating in elo_system.get_rankings()
        if "__" not in model
    ]
    
    # Generate category rankings
    category_rankings = {}
    for category, models in category_models.items():
        category_rankings[category] = [
            {"model": model.split("__")[0], "rating": elo_system.get_rating(f"{model}__{category}")}
            for model in models
        ]
        # Sort by rating (descending)
        category_rankings[category].sort(key=lambda x: x["rating"], reverse=True)
    
    return {
        "overall": overall_rankings,
        "categories": category_rankings
    }