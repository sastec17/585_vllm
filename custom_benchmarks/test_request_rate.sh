#!/bin/bash
# Test preemption values
# Example usage: ./test_request_rate.sh 

# Stop on errors
set -Eeuo pipefail

model=""
request_rates=()
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
        --rate|-r)
            request_rates+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            echo "    --model, -m       Specify the model name"
            echo "    --rate, -r      Specify desired request rate per second. Can specify more than once"    
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
    model="meta-llama/Llama-3.2-1B"
    echo "No model specified. Defaulting to $model"
fi

# Set default scripts if none provided
if [[ ${#request_rates[@]} -eq 0 ]]; then
    request_rates=(25 50 75 100 125 150)
    echo "No request rates provided. Defaulting to: ${request_rates[*]}"
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
mkdir -p "data/${sanitized_model}/request_rate/"
policies=("fcfs" "priority_round_robin_reverse")

for policy in "${policies[@]}"; do
    echo "Running online benchmarking for ${policy}..."
    # Runs server with RECOMPUTE for preemption
    # Start the model server in the background
    echo "Starting model server..."
    vllm serve $model \
                --disable-log-requests \
                --scheduling-policy $policy \
                --steps-before-preemption 5 \
                --enable-chunked-prefill=False \
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
        
    for rate in "${request_rates[@]}"; do
        # Run the benchmarking script
        echo "Running benchmarking script for ${policy} with ${rate} requests per second..."
        python3 online_benchmarking.py --backend vllm \
            --model "$model" \
            --schedule "$policy" \
            --dataset-path "$dataset_file" \
            --request-rate "$rate" \
            --percentile-metrics "ttft,tpot,itl,e2el" \
            --output-json "data/${sanitized_model}/request_rate/${rate}_${policy}_o.json"
        echo "Completed benchmarking script for ${policy} with ${rate} requests per second. Waiting to  send again..."
        sleep 5
    done
    # After the benchmarking script completes, stop the model server
    echo "Stopping the model server..."
    kill "$SERVER_PID"

    # Ensure the server process is terminated
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Model server stopped. Batch job completed."
done
echo "Test request rate completed"