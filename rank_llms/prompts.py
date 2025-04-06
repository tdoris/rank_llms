from typing import Dict, List
import random

def get_prompt_categories() -> List[str]:
    """Return the categories of prompts available for testing."""
    return [
        "General Knowledge",
        "Creative Writing",
        "Programming",
        "Reasoning",
        "Summarization"
    ]

def get_prompts_from_categories(categories: List[str], max_per_category: int = 10) -> Dict[str, List[str]]:
    """
    Get a dictionary of prompts by category.
    
    Args:
        categories: List of categories to include
        max_per_category: Maximum number of prompts to include per category
        
    Returns:
        Dictionary mapping category names to lists of prompts
    """
    all_prompts = {
        "General Knowledge": [
            "Explain the greenhouse effect and its role in climate change.",
            "What were the main causes and consequences of World War I?",
            "Describe the process of photosynthesis in plants.",
            "What is the difference between DNA and RNA?",
            "Explain the concept of supply and demand in economics.",
            "What are the main features of a parliamentary democracy?",
            "Describe the water cycle and its importance for Earth's ecosystems.",
            "What was the Renaissance and how did it influence European culture?",
            "Explain how vaccines work to protect against diseases.",
            "What are black holes and how do they form?"
        ],
        
        "Creative Writing": [
            "Write a short story about a time traveler who accidentally changes history.",
            "Create a poem about the changing of seasons.",
            "Write a dialogue between two strangers who meet on a delayed flight.",
            "Describe a futuristic city 100 years from now.",
            "Write a short mystery story where the detective is the culprit.",
            "Create a fairy tale that teaches children about honesty.",
            "Write a letter from an alien visitor describing Earth to their home planet.",
            "Describe a meal from the perspective of a food critic.",
            "Write a monologue from the perspective of a 100-year-old tree.",
            "Create a conversation between a river and the mountain it flows from."
        ],
        
        "Programming": [
            "Write a Python function to check if a string is a palindrome.",
            "Explain how to implement a binary search algorithm and its time complexity.",
            "Write a JavaScript function to calculate the Fibonacci sequence up to n terms.",
            "How would you design a simple user authentication system?",
            "Explain the difference between SQL and NoSQL databases with examples.",
            "Write a function to find the longest common subsequence of two strings.",
            "How would you implement a basic recommendation system?",
            "Explain the concept of dependency injection with an example.",
            "Write a function to determine if a binary tree is balanced.",
            "How would you design a URL shortening service?"
        ],
        
        "Reasoning": [
            "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "A farmer has 10 sheep, all but 9 die. How many sheep are left?",
            "You have two hourglasses, one that measures 4 minutes and one that measures 7 minutes. How can you measure exactly 9 minutes using only these two hourglasses?",
            "Three friends split the bill at a restaurant. They each pay $10 and give the waiter $30 in total. The bill comes to $25 so the waiter gives them $5 in change. Each person gets $1 back, so they've paid $9 each, totaling $27. The waiter kept $2. Where is the remaining $1?",
            "If you have a cake and cut it into 8 equal pieces with 3 straight cuts, how did you cut it?",
            "A doctor gives you 3 pills and tells you to take one every half hour. How long will the pills last?",
            "If two typists can type two pages in two minutes, how many typists would it take to type 18 pages in six minutes?",
            "A clock shows 3:15. What is the angle between the hour and minute hands?",
            "If a plane crashes exactly on the border between the US and Canada, where do they bury the survivors?"
        ],
        
        "Summarization": [
            "Summarize the plot of Shakespeare's 'Romeo and Juliet' in a few sentences.",
            "Explain the theory of relativity in simple terms.",
            "Summarize the key events of the American Civil War.",
            "What are the main principles of mindfulness meditation?",
            "Explain the core concepts of machine learning to a non-technical audience.",
            "Summarize the history of the internet from ARPANET to the present day.",
            "What are the key benefits and risks of artificial intelligence?",
            "Summarize how a democratic government functions.",
            "Explain the basic principles of quantum physics in simple terms.",
            "What are the main causes and effects of global warming?"
        ]
    }
    
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