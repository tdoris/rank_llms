#!/bin/bash
# test14b.sh - Comprehensive testing script for 14B-scale models on coding101 benchmark
# Generated $(date)

# Set up log directory
mkdir -p logs

# Set variables
PROMPTSET="coding101"
NUM_PROMPTS=5  # Number of prompts per category
LOG_FILE="logs/test14b_$(date +%Y%m%d_%H%M%S).log"

# Define models to compare
MODELS=(
  "cogito:14b"
  "gemma3:12b"
  "phi4:latest"
  "deepseek-r1:14b" 
)

# Ensure API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY is not set. Please set it before running this script."
  echo "Example: export ANTHROPIC_API_KEY=your-api-key"
  exit 1
fi

# Function to run a comparison between two models
run_comparison() {
  model_a=$1
  model_b=$2
  
  echo "===================================================="
  echo "Starting comparison between $model_a and $model_b"
  echo "Using promptset: $PROMPTSET with $NUM_PROMPTS prompts per category"
  echo "===================================================="
  
  python -m rank_llms compare "$model_a" "$model_b" \
    --promptset "$PROMPTSET" \
    --num-prompts "$NUM_PROMPTS" \
    --output "results/${model_a}__vs__${model_b}.md" \
    --log-level INFO
    
  echo "Comparison between $model_a and $model_b completed"
  echo "----------------------------------------------------"
}

# Create results directory
mkdir -p results

echo "Starting comprehensive model testing for coding101 benchmark" | tee -a "$LOG_FILE"
echo "Models to test: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run all possible comparisons
total_comparisons=$((${#MODELS[@]} * (${#MODELS[@]} - 1) / 2))
current=1

for ((i=0; i<${#MODELS[@]}; i++)); do
  for ((j=i+1; j<${#MODELS[@]}; j++)); do
    echo "Progress: Comparison $current of $total_comparisons" | tee -a "$LOG_FILE"
    
    model_a="${MODELS[$i]}"
    model_b="${MODELS[$j]}"
    
    echo "Running: $model_a vs $model_b" | tee -a "$LOG_FILE"
    run_comparison "$model_a" "$model_b" 2>&1 | tee -a "$LOG_FILE"
    
    echo "Completed comparison $current of $total_comparisons" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    ((current++))
  done
done

# Generate leaderboard after all comparisons are done
echo "Generating leaderboard for $PROMPTSET" | tee -a "$LOG_FILE"
python -m rank_llms leaderboard --promptset "$PROMPTSET" --force-refresh | tee -a "$LOG_FILE"

# Suggest additional tests
echo "" | tee -a "$LOG_FILE"
echo "Suggesting additional tests to improve confidence" | tee -a "$LOG_FILE"
python -m rank_llms suggest-tests --promptset "$PROMPTSET" --max-suggestions 5 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "All comparisons completed successfully" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Results are in the 'results' directory" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Print summary
echo "===================================================="
echo "Testing completed for all model pairs!"
echo "Total comparisons: $total_comparisons"
echo "Log file: $LOG_FILE"
echo "===================================================="