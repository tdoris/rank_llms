#!/usr/bin/env python3
import os
import time
import json
import pickle
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime

import typer
import ollama
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from anthropic import Anthropic
from pydantic import BaseModel, Field

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
    
    try:
        completion = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=300,
            temperature=0,
            system="Evaluate the quality of the given response. Return only valid JSON.",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        result = completion.content[0].text
        return json.loads(result)
    except Exception as e:
        console.print(f"[red]Error evaluating response: {e}")
        return {"score": 0, "feedback": f"Evaluation error: {str(e)}"}

def query_model(model: str, prompt: str) -> Dict[str, Any]:
    """Query a model using Ollama."""
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        return {
            "response": response.message.content,
            "response_time": end_time - start_time
        }
    except Exception as e:
        console.print(f"[red]Error querying model {model}: {e}")
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
    
    with open(archive_path, "wb") as f:
        pickle.dump(responses, f)
    
    console.print(f"[green]Archived results for {model} to {archive_path}")

def load_archived_results(model: str) -> List[ModelResponse]:
    """Load archived results for a model if they exist."""
    archive_path = get_archive_path(model)
    
    if archive_path.exists():
        with open(archive_path, "rb") as f:
            return pickle.load(f)
    
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
        
        for category, responses in by_category.items():
            f.write(f"### {category}\n\n")
            
            # Group by prompt
            by_prompt = {}
            for response in responses:
                if response.prompt not in by_prompt:
                    by_prompt[response.prompt] = []
                by_prompt[response.prompt].append(response)
            
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
    
    console.print(f"[green]Detailed results saved to {path.resolve()}")

@app.command()
def main(
    models: List[str] = typer.Argument(..., help="Models to test (must be available in Ollama)"),
    num_prompts: int = typer.Option(10, help="Number of prompts to test per category"),
    output: str = typer.Option("results.md", help="Output file for detailed results"),
    force_retest: bool = typer.Option(False, help="Force retesting of models even if archived results exist")
):
    """Test and rank LLM model responses using Ollama and Claude for evaluation."""
    # Check if ANTHROPIC_API_KEY is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set")
        raise typer.Exit(1)
    
    anthropic_client = Anthropic(api_key=api_key)
    
    # Check if models are available in Ollama
    available_models = {getattr(model, "model") for model in ollama.list().models}
    for model in models:
        if model not in available_models:
            console.print(f"[yellow]Warning: Model '{model}' not found in Ollama. Available models: {', '.join(available_models)}")
            if not typer.confirm("Continue anyway?"):
                raise typer.Exit(1)
    
    # Get prompts
    categories = get_prompt_categories()
    all_prompts = get_prompts_from_categories(categories, max_per_category=num_prompts)
    
    # Determine if this is a full test to use archived results
    full_test = is_full_test(all_prompts, categories)
    
    # Load archived results for models if this is a full test and we're not forcing a retest
    models_to_test = []
    loaded_responses = []
    
    for model in models:
        if full_test and not force_retest:
            archived_results = load_archived_results(model)
            if archived_results:
                console.print(f"[green]Loaded archived results for {model}")
                loaded_responses.extend(archived_results)
            else:
                models_to_test.append(model)
        else:
            models_to_test.append(model)
    
    results = EvaluationResult(models=models, responses=loaded_responses)
    
    # Only test models that need testing
    if models_to_test:
        total_tasks = len(models_to_test) * sum(len(prompts) for prompts in all_prompts.values())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("[cyan]Testing models...", total=total_tasks)
            
            for model in models_to_test:
                model_responses = []
                
                for category, prompts in all_prompts.items():
                    for prompt in prompts:
                        progress.update(task, description=f"[cyan]Testing {model} with {category} prompt")
                        
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
    
    # Save results
    save_results_markdown(results, output)
    
    # Display summary table
    console.print("\n[bold green]Results Summary:[/bold green]")
    summary_df = results.summary_dataframe()
    console.print(summary_df)

if __name__ == "__main__":
    app()