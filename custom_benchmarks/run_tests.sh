#!/bin/bash
# Run full test suite for specified model
# Example usage: ./run_tests.sh --model gpt2 --policy fcfs --script l --script tp 

# Stop on errors
set -Eeuo pipefail

model=""
policies=()
scripts=()

# Sanity check command line options
usage() {
  echo "Usage: $0 --model <model_name> [--policy <policy_name> ...] [--script <script_type> ...]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            model="$2"
            shift 2
            ;;
        --policy|-p)
            policies+=("$2")
            shift 2
            ;;
        --script|-s)
            scripts+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            echo "    --model, -m       Specify the model name"
            echo "    --policy, -p      Add scheduling policy from [fcfs, priority, priority_round_robin]. Defaults to all."
            echo "    --script, -s      Specify scripts to run from [l, tp, o]. Defaults to all."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Ensure model is specified
if [[ -z $model ]]; then
    echo "Error: --model is required."
    exit 1
fi

# Set default scripts if none provided
if [[ ${#scripts[@]} -eq 0 ]]; then
    scripts=("l" "tp" "o")
    echo "No script types provided. Defaulting to: ${scripts[*]}"
fi

# Set default policies if none provided
if [[ ${#policies[@]} -eq 0 ]]; then
    policies=("fcfs" "priority" "priority_round_robin")
    echo "No policies provided. Defaulting to: ${policies[*]}"
fi

# Check if custom dataset already exists for model
dataset_file="data/${model}_data.json"
if ! [ -e "$dataset_file" ]; then
    echo "Data doesn't exist for model ${model}. Creating now..."
    python3 data/create_dataset.py --model "$model"
fi

mkdir -p "data/${model}"
# Iterate over script types and policies
for script_type in "${scripts[@]}"; do
    for policy in "${policies[@]}"; do
        case $script_type in
            l)
                echo "Running latency script..."
                python3 benchmark_latency.py --input-json "$dataset_file" \
                    --model "$model" \
                    --scheduling-policy "$policy" \
                    --output-json "data/${model}/${model}_${policy}_latency.json"
                ;;
            tp)
                echo "Running throughput script..."
                python3 benchmark_throughput.py --dataset "$dataset_file" \
                    --model "$model" \
                    --scheduling-policy "$policy" \
                    --output-json "data/${model}/${model}_${policy}_throughput.json"
                ;;
            o)
                echo "Running online benchmarking..."
                MODEL_SERVER_CMD="vllm serve $model --swap-space 16 --disable-log-requests --scheduling-policy $policy"

                # Start the model server in the background
                echo "Starting model server..."
                $MODEL_SERVER_CMD &

                # Capture the process ID (PID) of the server
                SERVER_PID=$!
                echo "Model server started with PID: $SERVER_PID"

                # Wait for the server to be ready
                echo "Waiting for the model server to initialize - Use conservative value"
                sleep 120

                # Run the benchmarking script
                echo "Running benchmarking script..."
                python3 online_benchmarking.py --backend vllm \
                    --model "$model" \
                    --schedule "$policy" \
                    --dataset-path "$dataset_file"

                # After the benchmarking script completes, stop the model server
                echo "Stopping the model server..."
                kill "$SERVER_PID"

                # Ensure the server process is terminated
                wait "$SERVER_PID" 2>/dev/null || true
                echo "Model server stopped. Batch job completed."
                ;;
            *)
                echo "Unknown script type: $script_type"
                ;;
        esac
    done
done