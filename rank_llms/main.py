#!/usr/bin/env python3
import os
import time
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Set
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

from rank_llms.prompts import get_prompt_categories, get_prompts_from_categories

console = Console()
app = typer.Typer()

class ModelResponse(BaseModel):
    model: str
    prompt: str
    category: str
    response: str
    response_time: float
    score: Optional[float] = None
    feedback: Optional[str] = None
    timestamp: Optional[datetime] = None

class EvaluationResult(BaseModel):
    models: List[str]
    responses: List[ModelResponse]
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for response in self.responses:
            data.append({
                "model": response.model,
                "category": response.category,
                "prompt": response.prompt,
                "score": response.score,
                "response_time": response.response_time
            })
        return pd.DataFrame(data)
    
    def summary_dataframe(self) -> pd.DataFrame:
        df = self.to_dataframe()
        summary = df.groupby(["model", "category"])["score"].mean().reset_index()
        pivot = summary.pivot(index="model", columns="category", values="score")
        pivot["Overall"] = df.groupby("model")["score"].mean().values
        pivot["Response Time (s)"] = df.groupby("model")["response_time"].mean().values
        # Round the values
        return pivot.round(2).sort_values("Overall", ascending=False)

def evaluate_response(anthropic_client: Anthropic, prompt: str, response: str, category: str) -> Dict[str, Any]:
    """Evaluate a model's response using Claude."""
    evaluation_prompt = f"""You are evaluating the quality of an AI assistant's response to a user query. 
Rate the response on a scale from 1-10 where 1 is terrible and 10 is perfect.

Category: {category}

User Query: {prompt}

AI Assistant's Response: {response}

Provide your evaluation in the following JSON format:
{{
  "score": <score>,
  "feedback": "<brief explanation of why you gave this score>"
}}

Focus on accuracy, helpfulness, clarity, and appropriateness in your evaluation.
"""
    
    logger.info(f"Calling Anthropic API to evaluate response for category: {category}")
    try:
        completion = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=300,
            temperature=0,
            system="Evaluate the quality of the given response. Return only valid JSON.",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        result = completion.content[0].text
        logger.info(f"Received evaluation from Anthropic API")
        
        try:
            parsed_result = json.loads(result)
            logger.info(f"Evaluation score: {parsed_result.get('score', 'unknown')}")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Anthropic API: {e}")
            logger.error(f"Raw response: {result}")
            return {"score": 0, "feedback": f"JSON parsing error: {str(e)}"}
    except APIError as e:
        error_message = f"Anthropic API error: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        return {"score": 0, "feedback": error_message}
    except Exception as e:
        error_message = f"Error evaluating response: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")
        return {"score": 0, "feedback": f"Evaluation error: {str(e)}"}

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

def get_archive_path(model: str) -> Path:
    """Get the path to the archive file for a model."""
    return Path("test_archive") / f"{model.replace(':', '_').replace('/', '_')}.pkl"

def archive_results(model: str, responses: List[ModelResponse]):
    """Archive model test results to a file."""
    archive_path = get_archive_path(model)
    
    # Ensure the directory exists
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Archiving results for model {model} to {archive_path}")
    try:
        with open(archive_path, "wb") as f:
            pickle.dump(responses, f)
        
        logger.info(f"Successfully archived {len(responses)} responses for model {model}")
        console.print(f"[green]Archived results for {model} to {archive_path}")
    except Exception as e:
        error_message = f"Error archiving results for {model}: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")

def load_archived_results(model: str) -> List[ModelResponse]:
    """Load archived results for a model if they exist."""
    archive_path = get_archive_path(model)
    
    if archive_path.exists():
        logger.info(f"Loading archived results for model {model} from {archive_path}")
        try:
            with open(archive_path, "rb") as f:
                results = pickle.load(f)
            
            logger.info(f"Successfully loaded {len(results)} archived responses for model {model}")
            return results
        except Exception as e:
            error_message = f"Error loading archived results for {model}: {e}"
            logger.error(error_message)
            console.print(f"[red]{error_message}")
            return []
    
    logger.info(f"No archived results found for model {model}")
    return []

def is_full_test(all_prompts: Dict[str, List[str]], categories: List[str]) -> bool:
    """Determine if this is a full test with all prompts."""
    for category in categories:
        # If any category has less than 10 prompts, it's not a full test
        if len(all_prompts.get(category, [])) < 10:
            return False
    
    return True

def save_results_markdown(results: EvaluationResult, output_path: str = "results.md"):
    """Save detailed results to a markdown file."""
    path = Path(output_path)
    
    logger.info(f"Saving detailed results to {path}")
    try:
        with open(path, "w") as f:
            f.write("# LLM Evaluation Results\n\n")
            
            # Write summary table
            f.write("## Summary\n\n")
            summary_df = results.summary_dataframe()
            f.write(summary_df.to_markdown() + "\n\n")
            
            # Write detailed results by category
            f.write("## Detailed Results\n\n")
            
            # Group responses by category
            by_category = {}
            for response in results.responses:
                if response.category not in by_category:
                    by_category[response.category] = []
                by_category[response.category].append(response)
            
            logger.info(f"Writing detailed results for {len(by_category)} categories")
            for category, responses in by_category.items():
                f.write(f"### {category}\n\n")
                
                # Group by prompt
                by_prompt = {}
                for response in responses:
                    if response.prompt not in by_prompt:
                        by_prompt[response.prompt] = []
                    by_prompt[response.prompt].append(response)
                
                logger.info(f"Writing {len(by_prompt)} prompts for category {category}")
                for prompt_idx, (prompt, prompt_responses) in enumerate(by_prompt.items()):
                    f.write(f"**Prompt {prompt_idx+1}**: {prompt}\n\n")
                    
                    # Sort responses by score (descending)
                    prompt_responses.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
                    
                    for response in prompt_responses:
                        f.write(f"*Model: {response.model} (Score: {response.score}, Time: {response.response_time:.2f}s)*\n\n")
                        f.write(f"```\n{response.response}\n```\n\n")
                        if response.feedback:
                            f.write(f"**Feedback**: {response.feedback}\n\n")
                        if response.timestamp:
                            f.write(f"*Tested: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    
                    f.write("---\n\n")
        
        logger.info(f"Successfully saved detailed results to {path.resolve()}")
        console.print(f"[green]Detailed results saved to {path.resolve()}")
    except Exception as e:
        error_message = f"Error saving results to {path}: {e}"
        logger.error(error_message)
        console.print(f"[red]{error_message}")

@app.command()
def main(
    models: List[str] = typer.Argument(..., help="Models to test (must be available in Ollama)"),
    num_prompts: int = typer.Option(10, help="Number of prompts to test per category"),
    output: str = typer.Option("results.md", help="Output file for detailed results"),
    force_retest: bool = typer.Option(False, help="Force retesting of models even if archived results exist"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
):
    """Test and rank LLM model responses using Ollama and Claude for evaluation."""
    # Set log level based on command line argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}")
        raise typer.Exit(1)
    
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")
    
    logger.info(f"Starting LLM ranking with models: {models}, num_prompts: {num_prompts}")
    
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
    
    # Get prompts
    logger.info("Getting prompt categories and prompts")
    categories = get_prompt_categories()
    all_prompts = get_prompts_from_categories(categories, max_per_category=num_prompts)
    logger.info(f"Selected {sum(len(prompts) for prompts in all_prompts.values())} prompts across {len(categories)} categories")
    
    # Determine if this is a full test to use archived results
    full_test = is_full_test(all_prompts, categories)
    logger.info(f"Test type: {'Full test' if full_test else 'Partial test'}")
    
    # Load archived results for models if this is a full test and we're not forcing a retest
    models_to_test = []
    loaded_responses = []
    
    for model in models:
        if full_test and not force_retest:
            archived_results = load_archived_results(model)
            if archived_results:
                logger.info(f"Using archived results for model {model}")
                console.print(f"[green]Loaded archived results for {model}")
                loaded_responses.extend(archived_results)
            else:
                logger.info(f"No archived results found for model {model}, will test")
                models_to_test.append(model)
        else:
            if force_retest:
                logger.info(f"Force retest enabled, will test model {model}")
            else:
                logger.info(f"Partial test, will test model {model}")
            models_to_test.append(model)
    
    results = EvaluationResult(models=models, responses=loaded_responses)
    
    # Only test models that need testing
    if models_to_test:
        total_tasks = len(models_to_test) * sum(len(prompts) for prompts in all_prompts.values())
        logger.info(f"Beginning testing of {len(models_to_test)} models with {total_tasks} total tasks")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("[cyan]Testing models...", total=total_tasks)
            
            for model in models_to_test:
                logger.info(f"Starting tests for model: {model}")
                model_responses = []
                
                for category, prompts in all_prompts.items():
                    logger.info(f"Testing category: {category} with {len(prompts)} prompts")
                    for prompt in prompts:
                        progress.update(task, description=f"[cyan]Testing {model} with {category} prompt")
                        logger.info(f"Testing prompt: {prompt[:50]}...")
                        
                        # Query model
                        model_result = query_model(model, prompt)
                        
                        # Evaluate response
                        evaluation = evaluate_response(
                            anthropic_client, 
                            prompt, 
                            model_result["response"], 
                            category
                        )
                        
                        # Create response object
                        response = ModelResponse(
                            model=model,
                            prompt=prompt,
                            category=category,
                            response=model_result["response"],
                            response_time=model_result["response_time"],
                            score=evaluation.get("score"),
                            feedback=evaluation.get("feedback"),
                            timestamp=datetime.now()
                        )
                        
                        # Add to model responses and results
                        model_responses.append(response)
                        results.responses.append(response)
                        
                        # Update progress
                        progress.update(task, advance=1)
                
                # Archive results if this is a full test
                if full_test:
                    archive_results(model, model_responses)
    else:
        logger.info("No models need testing, using only archived results")
    
    # Save results
    logger.info(f"Saving results to {output}")
    save_results_markdown(results, output)
    
    # Display summary table
    logger.info("Generating summary table")
    console.print("\n[bold green]Results Summary:[/bold green]")
    summary_df = results.summary_dataframe()
    console.print(summary_df)
    
    logger.info("Testing complete")
    return summary_df

if __name__ == "__main__":
    app()