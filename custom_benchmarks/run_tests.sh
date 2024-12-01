#!/bin/bash
# Run full test suite for specified model
# Example usage: ./run_tests.sh --model gpt2 --output-len 1024 --policy fcfs --script l --script tp 

# Stop on errors
set -Eeuo pipefail

model=""
output_length=512
policies=()
scripts=()
num_preempt=60

# Sanity check command line options
usage() {
  echo "Usage: $0 --model <model_name> --output-len <output_length> --num-preempt <num_preempt_tokens> [--policy <policy_name> ...] [--script <script_type> ...]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            model="$2"
            shift 2
            ;;
        --num-preempt|-n)
            num_preempt="$2"
            shift 2
            ;;
        --output-len|-o)
            output_length="$2"
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

sanitized_model="${model//\//_}"
mkdir -p "data/${sanitized_model}/${output_length}"
# Check if custom dataset already exists for model
dataset_file="data/${sanitized_model}/${output_length}/${sanitized_model}_data.json"

if ! [ -e "$dataset_file" ]; then
    echo "Data doesn't exist for model ${model}. Creating now..."
    python3 data/create_dataset.py --model "$model" --output-len "$output_length"
fi

# Iterate over script types and policies
for script_type in "${scripts[@]}"; do
    for policy in "${policies[@]}"; do
        # Specify preemption with priority_round_robin
        if [[ "$policy" == "priority_round_robin" ]]; then
            preempt_flag=f"--steps-before-preemption ${num_preempt}"
        else
            preempt_flag=""
        fi
        case $script_type in
            l)
                echo "Running latency script for ${policy}..."
                python3 benchmark_latency.py --input-json "$dataset_file" \
                    --model "$model" \
                    --scheduling-policy "$policy" \
                    $preempt_flag \
                    --output-len "$output_length" \
                    --disable_async_output_proc \
                    --output-json "data/${sanitized_model}/${output_length}/${policy}_l.json"
                ;;
            tp)
                echo "Running throughput script for ${policy}..."
                python3 benchmark_throughput.py --dataset "$dataset_file" \
                    --model "$model" \
                    --scheduling-policy "$policy" \
                    $preempt_flag \
                    --output-len "$output_length" \
                    --disable_async_output_proc \
                    --output-json "data/${sanitized_model}/${output_length}/${policy}_tp.json"
                ;;
            o)
                echo "Running online benchmarking for ${policy}..."
                # Runs server with RECOMPUTE for preemption
                # MODEL_SERVER_CMD="vllm serve $model --disable-log-requests --scheduling-policy $policy $preempt_flag --disable-async-output-proc --output-len "$output_length""

                # Start the model server in the background
                echo "Starting model server..."
                vllm serve $model \
                            --disable-log-requests \
                            --scheduling-policy $policy \
                            $preempt_flag \
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
                    --output-json "data/${sanitized_model}/${output_length}/${policy}_o.json"
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