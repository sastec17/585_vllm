#!/bin/bash
# Test preemption values
# Example usage: ./run_tests.sh --output-len 1024

# Stop on errors
set -Eeuo pipefail

model=""
preempt_tokens=()
output_length=1024

# Sanity check command line options
usage() {
  echo "Usage: $0 --model <model_name> --output-len <output_length> "
}

# Process command line
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            model="$2"
            shift 2
            ;;
        --output-len|-o)
            output_length="$2"
            shift 2
            ;;
        --token|-t)
            preempt_tokens+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            echo "    --model, -m       Specify the model name"
            echo "    --token, -t       Specify desired preempt token val. Can specify more than once"    
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Check if model is specified
if [[ -z $model ]]; then
    model="facebook/opt-125m"
    echo "No model specified. Defaulting to $model"
fi

# Set default scripts if none provided
if [[ ${#preempt_tokens[@]} -eq 0 ]]; then
    preempt_tokens=(75 100)
    echo "No script types provided. Defaulting to: ${preempt_tokens[*]}"
fi

# Ensure dataset file is created
sanitized_model="${model//\//_}"
mkdir -p "data/${sanitized_model}/${output_length}"
# Check if custom dataset already exists for model
dataset_file="data/${sanitized_model}/${output_length}/${sanitized_model}_data.json"

if ! [ -e "$dataset_file" ]; then
    echo "Data doesn't exist for model ${model}. Creating now..."
    python3 data/create_dataset.py --model "$model" --output-len "$output_length"
fi

policy="priority_round_robin"
for preempt_val in "${preempt_tokens[@]}"; do
    echo "Running online benchmarking for ${policy} with ${preempt_val} magic tokens..."
    # Runs server with RECOMPUTE for preemption
    # Start the model server in the background
    echo "Starting model server..."
    vllm serve $model \
                --disable-log-requests \
                --scheduling-policy $policy \
                --steps-before-preemption $preempt_val \
                --disable-async-output-proc &

    # Capture the process ID (PID) of the server
    SERVER_PID=$!
    echo "Model server started with PID: $SERVER_PID"

    echo "Waiting for the model server to initialize. Set timeout limit"
    max_attempts=30
    attempt=0
    server_ready=false
    while [[ $attempt -lt $max_attempts ]]; do
        if curl --silent --fail http://localhost:8000/health; then
            server_ready=true
            break
        fi
        echo "Server not ready yet. Retrying in 5 seconds..."
        sleep 5
        attempt=$((attempt + 1))
    done

    if [[ $server_ready == false ]]; then
        echo "Error: Server did not become ready within the timeout period."
        kill "$SERVER_PID"
        exit 1
    fi
    echo "Server is ready."

    # Run the benchmarking script
    echo "Running benchmarking script for ${policy}..."
    python3 online_benchmarking.py --backend vllm \
        --model "$model" \
        --schedule "$policy" \
        --dataset-path "$dataset_file" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --output-json "data/${sanitized_model}/test_preempt/preempt_${preempt_val}_o.json"
    # After the benchmarking script completes, stop the model server
    echo "Stopping the model server..."
    kill "$SERVER_PID"

    # Ensure the server process is terminated
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Model server stopped. Batch job completed."
done

# construct e2el and graphs
mkdir -p graphs/preempt
python3 graphs/preempt.py "${preempt_tokens[@]}"