"""ELO rating system for ranking LLM models based on comparative evaluations."""

import math
from typing import Dict, List, Tuple, Set, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger("rank_llms")

# Default K-factor determines how quickly ratings change
DEFAULT_K = 32
# Default starting ELO rating
DEFAULT_STARTING_ELO = 1400

class EloRatingSystem:
    """
    Implements an ELO rating system for ranking LLM models based on head-to-head comparisons.
    """
    
    def __init__(self, k_factor: int = DEFAULT_K, starting_elo: int = DEFAULT_STARTING_ELO, promptset: str = "basic1"):
        self.k_factor = k_factor
        self.starting_elo = starting_elo
        self.promptset = promptset
        self.ratings: Dict[str, float] = {}
        self.match_history: List[Dict] = []
    
    def get_rating(self, model: str) -> float:
        """Get the current ELO rating for a model. If not rated yet, returns the starting ELO."""
        return self.ratings.get(model, self.starting_elo)
    
    def expected_score(self, model_a: str, model_b: str) -> float:
        """Calculate the expected score (probability of winning) for model_a against model_b."""
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    def update_ratings(self, model_a: str, model_b: str, score_a: float, category: Optional[str] = None) -> Tuple[float, float]:
        """
        Update the ELO ratings after a match between model_a and model_b.
        
        Args:
            model_a: First model
            model_b: Second model
            score_a: Actual score for model_a (1.0 for win, 0.5 for draw, 0.0 for loss)
            category: Optional category where the match was played
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Get current ratings
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)
        
        # Calculate expected scores
        expected_a = self.expected_score(model_a, model_b)
        expected_b = 1.0 - expected_a
        
        # Calculate new ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1.0 - score_a) - expected_b)
        
        # Update ratings
        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b
        
        # Record match
        self.match_history.append({
            "model_a": model_a,
            "model_b": model_b,
            "old_rating_a": rating_a,
            "old_rating_b": rating_b,
            "new_rating_a": new_rating_a,
            "new_rating_b": new_rating_b,
            "score_a": score_a,
            "category": category
        })
        
        return (new_rating_a, new_rating_b)
    
    def register_match_result(self, model_a: str, model_b: str, wins_a: int, wins_b: int, 
                             draws: int = 0, category: Optional[str] = None) -> None:
        """
        Register the result of multiple comparisons between two models.
        
        Args:
            model_a: First model
            model_b: Second model
            wins_a: Number of times model_a was ranked higher
            wins_b: Number of times model_b was ranked higher
            draws: Number of times the models were ranked equally
            category: Optional category where the comparisons were made
        """
        total_matches = wins_a + wins_b + draws
        if total_matches == 0:
            logger.warning(f"Attempted to register match with zero comparisons between {model_a} and {model_b}")
            return
        
        # Calculate the aggregate score for model_a
        score_a = (wins_a + 0.5 * draws) / total_matches
        
        # Update the ratings
        self.update_ratings(model_a, model_b, score_a, category)
        
        logger.info(f"Registered match result: {model_a} vs {model_b} = {wins_a}-{wins_b}-{draws} in {category or 'all categories'}")
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get a sorted list of (model, rating) tuples, from highest to lowest rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
    
    def get_all_models(self) -> Set[str]:
        """Get the set of all models in the system."""
        return set(self.ratings.keys())
    
    def save_ratings(self, file_path: str) -> None:
        """Save the current ratings to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "ratings": self.ratings,
            "k_factor": self.k_factor,
            "starting_elo": self.starting_elo,
            "promptset": self.promptset,
            "match_history": self.match_history
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved ELO ratings to {path}")
    
    @classmethod
    def load_ratings(cls, file_path: str, promptset: str = "basic1") -> 'EloRatingSystem':
        """Load ratings from a file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Ratings file {path} does not exist, creating new ELO system with promptset '{promptset}'")
            return cls(promptset=promptset)
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Use the promptset from the file if available, otherwise use the provided one
            file_promptset = data.get('promptset', promptset)
            
            elo_system = cls(
                k_factor=data.get('k_factor', DEFAULT_K),
                starting_elo=data.get('starting_elo', DEFAULT_STARTING_ELO),
                promptset=file_promptset
            )
            elo_system.ratings = data.get('ratings', {})
            elo_system.match_history = data.get('match_history', [])
            
            logger.info(f"Loaded ELO ratings from {path} with {len(elo_system.ratings)} models for promptset '{elo_system.promptset}'")
            return elo_system
        
        except Exception as e:
            logger.error(f"Error loading ratings from {path}: {e}")
            return cls()