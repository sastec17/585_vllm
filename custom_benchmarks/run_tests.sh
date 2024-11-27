#!/bin/bash
# Run full test suite for specified model

# stop on errors
set -Eeuo pipefail

model=""
policies=()
# Sanity check command line options
usage() {
  echo "Usage: $0 --model <model_name> [--policy <policy_name> ...]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            model="$2"
            shift 2 # Shift past argument and its value
            ;;
        --policy|-p)
            policies+=("$2")
            shift 2 # Shift past argument and its value
            ;;
        --help|-h)
            echo "Usage: $0 --model <model_name> [--policy <policy_name> ...]"
            echo "    --model, -m       Specify the model name"
            echo "    --policy, -p      Add scheduling policy (can be specified multiple times)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# model must be specified
if [[ -z $model ]]; then
    echo "Error: --model is required."
    exit 1
fi

# check if custom dataset already exists for model
dataset_file="data/${model}_data.json"
if ! [ -e "$dataset_file" ]; then
    echo "Data doesn't exist for model ${model}. Creating now..."
    python3 data/create_dataset.py --model "$model"
fi

# TODO: Iterate over policies passed in + run accordingly
echo "Running latency script..."
python3 benchmark_latency.py --input-json data/"$model"_data.json \
    --model "$model" \
    --scheduling-policy fcfs \
    --output-json data/"$model"_priority_latency.json

echo "Running throughput script..."
python3 benchmark_throughput.py --dataset data/gpt2_data.json \
    --model "$model" \
    --scheduling-policy fcfs \
    --output-json data/"$model"_fcfs_throughput.json

echo "Running online benchmarking..."
MODEL_SERVER_CMD="vllm serve $model --swap-space 16 --disable-log-requests --scheduling-policy fcfs"

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
    --model $model \
    --schedule fcfs \
    --dataset-path data/"$model"_data.json

# After the benchmarking script completes, stop the model server
echo "Stopping the model server..."
kill $SERVER_PID

# Ensure the server process is terminated
wait $SERVER_PID 2>/dev/null
echo "Model server stopped. Batch job completed."