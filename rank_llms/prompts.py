from typing import Dict, List
import random
import json
import os
from pathlib import Path
import logging

# Get logger
logger = logging.getLogger("rank_llms")

def get_promptset_path(promptset_name: str) -> Path:
    """
    Get the path to a promptset JSON file.
    
    Args:
        promptset_name: Name of the promptset (without .json extension)
        
    Returns:
        Path to the promptset file
    """
    # Check if the name already includes .json extension
    if not promptset_name.endswith('.json'):
        promptset_name = f"{promptset_name}.json"
    
    # Check if the path is absolute or relative
    if os.path.isabs(promptset_name):
        return Path(promptset_name)
    else:
        return Path("promptsets") / promptset_name

def load_promptset(promptset_name: str = "basic1") -> Dict[str, List[str]]:
    """
    Load a promptset from a JSON file.
    
    Args:
        promptset_name: Name of the promptset file (without .json extension)
        
    Returns:
        Dictionary of prompts by category
    """
    promptset_path = get_promptset_path(promptset_name)
    
    logger.info(f"Loading promptset from {promptset_path}")
    try:
        with open(promptset_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading promptset {promptset_name}: {e}")
        raise

def get_prompt_categories(promptset_name: str = "basic1") -> List[str]:
    """
    Return the categories of prompts available for testing in the specified promptset.
    
    Args:
        promptset_name: Name of the promptset to use
        
    Returns:
        List of prompt categories
    """
    promptset = load_promptset(promptset_name)
    return list(promptset.keys())

def get_prompts_from_categories(
    categories: List[str], 
    max_per_category: int = 10,
    promptset_name: str = "basic1"
) -> Dict[str, List[str]]:
    """
    Get a dictionary of prompts by category from the specified promptset.
    
    Args:
        categories: List of categories to include
        max_per_category: Maximum number of prompts to include per category
        promptset_name: Name of the promptset to use
        
    Returns:
        Dictionary mapping category names to lists of prompts
    """
    all_prompts = load_promptset(promptset_name)
    
    result = {}
    for category in categories:
        if category in all_prompts:
            prompts = all_prompts[category]
            # Take a random sample if we need fewer than all prompts
            if max_per_category < len(prompts):
                result[category] = random.sample(prompts, max_per_category)
            else:
                result[category] = prompts
    
    return result