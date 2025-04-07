#!/usr/bin/env python3
"""
Test script for the promptset functionality.
"""

import os
import logging
from pathlib import Path
import json
from rank_llms.prompts import load_promptset, get_prompt_categories, get_prompts_from_categories

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tester")

def test_promptsets():
    """Test the promptset functionality"""
    # Test loading the default promptset
    logger.info("Testing default promptset loading")
    promptset = load_promptset()
    
    logger.info(f"Loaded default promptset with {len(promptset)} categories")
    for category, prompts in promptset.items():
        logger.info(f"  - {category}: {len(prompts)} prompts")
    
    # Test getting categories
    logger.info("\nTesting get_prompt_categories")
    categories = get_prompt_categories()
    logger.info(f"Categories: {categories}")
    
    # Test getting prompts from categories
    logger.info("\nTesting get_prompts_from_categories")
    prompts = get_prompts_from_categories(categories, max_per_category=2)
    for category, cat_prompts in prompts.items():
        logger.info(f"Category: {category}")
        for i, prompt in enumerate(cat_prompts, 1):
            truncated = prompt[:50] + "..." if len(prompt) > 50 else prompt
            logger.info(f"  {i}. {truncated}")
    
    # Create a new test promptset
    logger.info("\nCreating a test promptset")
    test_promptset = {
        "Test Category 1": [
            "This is prompt 1 in category 1",
            "This is prompt 2 in category 1"
        ],
        "Test Category 2": [
            "This is prompt 1 in category 2",
            "This is prompt 2 in category 2"
        ]
    }
    
    test_file = Path("promptsets") / "test.json"
    with open(test_file, "w") as f:
        json.dump(test_promptset, f, indent=2)
    
    logger.info(f"Created test promptset at {test_file}")
    
    # Test loading the test promptset
    logger.info("\nTesting loading of test promptset")
    test_loaded = load_promptset("test")
    
    logger.info(f"Loaded test promptset with {len(test_loaded)} categories")
    for category, prompts in test_loaded.items():
        logger.info(f"  - {category}: {len(prompts)} prompts")
    
    # Test getting categories from test promptset
    logger.info("\nTesting get_prompt_categories with test promptset")
    test_categories = get_prompt_categories("test")
    logger.info(f"Categories: {test_categories}")
    
    # Test getting prompts from categories with test promptset
    logger.info("\nTesting get_prompts_from_categories with test promptset")
    test_prompts = get_prompts_from_categories(test_categories, max_per_category=2, promptset_name="test")
    for category, cat_prompts in test_prompts.items():
        logger.info(f"Category: {category}")
        for i, prompt in enumerate(cat_prompts, 1):
            logger.info(f"  {i}. {prompt}")
    
    logger.info("\nAll tests passed!")

if __name__ == "__main__":
    test_promptsets()