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
            "Compare and contrast the theories of dark matter and modified gravity (MOND) as explanations for galactic rotation curves, discussing the observational evidence that supports each theory.",
            "Analyze the historical development and evolution of central banking systems from the 17th century to present day, examining how their roles have changed across different economic paradigms.",
            "Explain the chemical mechanisms of neurotransmitter action in synaptic transmission, including the role of various receptor types and how psychoactive drugs modify these processes.",
            "Describe the mathematical foundations of quantum field theory and how it reconciles quantum mechanics with special relativity, explaining the significance of the Dirac equation.",
            "Discuss the geological evidence for the Snowball Earth hypothesis and evaluate competing theories about Earth's climate during the Neoproterozoic era.",
            "Compare the different interpretations of quantum mechanics (Copenhagen, Many-Worlds, Pilot Wave, etc.) and analyze the philosophical implications of each regarding the nature of reality.",
            "Analyze how modern monetary theory challenges conventional economic thinking about government spending, taxation, and inflation in sovereign currency-issuing nations.",
            "Explain the evolutionary and ecological factors that have contributed to the development of eusociality in certain insect species but not others, providing specific examples.",
            "Discuss the historical development of linguistic relativism from the Sapir-Whorf hypothesis to current research, examining evidence both for and against the theory that language shapes cognition.",
            "Analyze the biochemical pathways involved in cellular autophagy and explain its significance in normal cellular functioning, aging, and disease states."
        ],
        
        "Creative Writing": [
            "Write a short story from the perspective of a quantum particle experiencing wave-particle duality during the double-slit experiment. Make it philosophically engaging while being scientifically accurate.",
            "Compose a polyphonic poem with three distinct voices: an AI researcher in 2045, a sentient AI they've created, and a historian from 2145 looking back on their interaction. Each voice should have a distinct style and perspective on consciousness.",
            "Write a dialogue between Socrates and a modern climate scientist, in which Socrates uses his method of questioning to explore the ethical dimensions of climate change responsibility across generations.",
            "Create a short story set 500 years in the future that explores the social implications of extreme human longevity (800+ year lifespans). Focus on memory, identity, and relationship dynamics.",
            "Write a microfiction piece (under 500 words) that uses a non-linear narrative structure to tell the story of a significant historical event from the perspectives of three different observers, separated by time but connected through a single object.",
            "Compose a fairy tale for adults that reinterprets a classic philosophical thought experiment (e.g., the Ship of Theseus, the Chinese Room, or the Trolley Problem) as a magical journey with metaphorical characters and settings.",
            "Write a monologue from the perspective of the last speaker of a dying language, weaving in untranslatable concepts from that language that reveal a unique worldview being lost. Create some linguistically plausible untranslatable terms.",
            "Compose a story that employs the literary device of magical realism to explore the psychological experience of synesthesia or another neurological phenomenon. Use sensory descriptions that blend perceptions in surprising but internally consistent ways.",
            "Create a short speculative fiction piece about a society that has developed an alternative economic system not based on money or barter but on a novel concept you'll invent. Explain how everyday transactions work in this system.",
            "Write a narrative that shifts between three time periods (ancient past, present, and far future) connected by a specific geographical location, showing how the meaning and significance of this place changes while certain patterns remain constant."
        ],
        
        "Programming": [
            "Design and implement a thread-safe LRU (Least Recently Used) cache in Python with a specified capacity that handles concurrent access. Include proper synchronization mechanisms and discuss the performance implications of your design choices.",
            "Write a recursive algorithm to find all possible valid colorings of a graph with m colors such that no adjacent vertices have the same color. Analyze its time complexity and provide an optimization to improve performance on large graphs.",
            "Implement a system for detecting and preventing race conditions in a distributed database that handles concurrent transactions. Explain your approach to deadlock detection and prevention.",
            "Create a custom parser for a simplified query language that allows filtering, aggregation, and joining of datasets. Your implementation should handle parsing, validation, and execution of queries with proper error handling.",
            "Design a system for real-time anomaly detection in streaming time-series data. Implement an algorithm that adapts its thresholds automatically as data patterns evolve, explaining how it handles different types of anomalies.",
            "Implement a memory-efficient trie data structure for autocomplete suggestions that can handle millions of strings. Include functionality to return the top N suggestions based on frequency and edit distance from the input.",
            "Write an algorithm to detect cycles in a directed graph and classify them according to their properties. Implement a visualization method that highlights different types of cycles with different colors.",
            "Design and implement a backpressure mechanism for a system that processes messages from multiple producers at varying rates. Your solution should prevent memory overflow while maximizing throughput.",
            "Implement a custom garbage collection algorithm for a simple programming language runtime. Explain how it identifies unreachable objects and handles memory fragmentation.",
            "Create a bloom filter implementation with dynamic scaling that maintains a target false positive rate as the number of elements grows. Analyze the space-time tradeoffs of your approach."
        ],
        
        "Reasoning": [
            "You have 12 identical-looking coins, one of which is counterfeit and has a slightly different weight (you don't know if it's heavier or lighter). Using a balance scale, how can you identify the counterfeit coin and determine whether it's heavier or lighter in just 3 weighings? Explain your reasoning step by step.",
            "Consider an infinite chess board with a knight placed on a square. If the knight makes random moves (each of the 8 possible moves having equal probability whenever all 8 are available), what is the probability that the knight will return to its starting square at some point? Provide a mathematical argument.",
            "In a room of just 23 people, what is the probability that at least two people share the same birthday? Explain your calculation and why this probability is much higher than most people intuitively expect.",
            "You have two identical glass spheres. You're in a 100-story building, and you need to determine the highest floor from which the spheres can be dropped without breaking. What strategy minimizes the worst-case number of drops needed? Analyze the problem mathematically.",
            "Six people are at a party. Prove that either at least three of them all know each other, or at least three of them all don't know each other. (Assume that 'knowing' is a symmetric relation: if A knows B, then B knows A.) Use graph theory in your explanation.",
            "You have 100 prisoners numbered 1 to a 100, and 100 boxes also numbered 1 to 100. In each box, there is a slip of paper with a prisoner's number on it. Each prisoner is allowed to open up to 50 boxes to try to find their own number. The prisoners will all be freed if every prisoner finds their number. The prisoners can devise a strategy beforehand, but once the searching begins, they cannot communicate. What strategy gives them the best chance of being freed, and what is that probability?",
            "Consider an ant walking on the edges of a cube, starting at one vertex. At each vertex, the ant randomly chooses one of the three connected edges to walk along next. What is the expected number of edges the ant will walk before returning to the starting vertex? Show your work.",
            "Four people need to cross a bridge at night. They have one flashlight, and the bridge can only support two people at once. The four people take 1, 2, 5, and 10 minutes respectively to cross the bridge. When two people cross together, they move at the speed of the slower person. What is the minimum time needed for all four people to cross the bridge? Prove that your solution is optimal.",
            "Consider a system where you have 5 processes and 3 resources. Each process may claim up to 1 unit of each resource type, and there are 2 units of each resource type available. Is this system guaranteed to be deadlock-free? Provide a proof or counterexample.",
            "There are 7 bridge crossing points over a river that runs east to west. There are 5 villages on the north side and 5 on the south side of the river. What is the minimum number of crossings needed for everyone to be able to travel from any village to any other village if bridges can be placed anywhere? Justify your answer using graph theory."
        ],
        
        "Summarization": [
            "Summarize Walter Isaacson's biography of Einstein, highlighting the key scientific contributions, personal relationships, and historical context that shaped Einstein's work and legacy. Include an analysis of how Einstein's scientific worldview evolved throughout his life.",
            "Provide a comprehensive summary of the competing theories about the origin of consciousness, including higher-order theories, global workspace theory, integrated information theory, and quantum theories. Explain the key evidence supporting each approach and major criticisms.",
            "Summarize the history, key findings, and current status of string theory in theoretical physics. Explain its attempts to unify quantum mechanics and general relativity, major breakthroughs, challenges, and criticisms from within the physics community.",
            "Provide a detailed summary of the 2008 global financial crisis, including its causes, the sequence of major events, policy responses, and lasting economic and regulatory consequences. Include perspectives from different economic schools of thought.",
            "Summarize the current scientific understanding of how memory works in the human brain, from molecular mechanisms to systems-level organization. Include explanations of different memory types, encoding processes, retrieval mechanisms, and common memory failures.",
            "Provide a comprehensive summary of the development of artificial general intelligence research from its early philosophical foundations to current approaches. Analyze the key technical challenges, competing methodologies, and major ethical considerations.",
            "Summarize the evolution of modern encryption from World War II to present day, explaining how encryption algorithms work in principle, major historical developments, the impact of quantum computing, and current debates about encryption policy.",
            "Provide a detailed analysis of the human microbiome's role in health and disease, summarizing recent research on how microbial communities influence immune function, metabolism, brain development, and potential therapeutic applications.",
            "Summarize the key debates in contemporary cosmology, including the nature of dark matter and dark energy, the multiverse hypothesis, competing models of cosmic inflation, and approaches to resolving tensions in measurements of the Hubble constant.",
            "Provide a comprehensive summary of how climate feedback mechanisms work, including both positive and negative feedbacks, their relative strengths, tipping points, and how they're represented in climate models. Explain current scientific uncertainties."
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