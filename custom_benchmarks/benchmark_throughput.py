"""
Benchmark the throughput of processing a fixed number of requests.

Use Oracle w/ output lengths to establish priority

Example usage:
python3 custom_benchmarks/benchmark_throughput.py \
    --dataset data/gpt2_data.json \
    --model gpt2 \
    --scheduling-policy priority_round_robin \
    --output-json data/gpt2_rr_throughput.json

Inspired by benchmarks/benchmark_throughput.py
"""
import argparse
import dataclasses
import json
import random
import time
from typing import List, Tuple

from transformers import (PreTrainedTokenizerBase)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser

def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace):
    dataset_path: str = args.dataset
    num_requests: int = args.num_prompts
    # can add fixed output_len
    model: str = args.model
    # Load dataset
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
    random.shuffle(dataset)
#     * Get prompt / completion
#     * Tokenize + prune + append to filtered_dataset

    filtered_dataset: List[Tuple[str, int]] = []
    for i in range((len(dataset))):
        if len(filtered_dataset) == num_requests:
            break
        # tokenize prompt inputs to filter
        prompt = dataset[i]['input']
        prompt_token_ids = tokenizer(prompt).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = dataset[i]['output_tokens']
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # priority = output_len
        filtered_dataset.append((prompt, prompt_len, output_len))
    return filtered_dataset

# # TODO: Ensure llm loaded properly for each mode
def run_vllm(
        requests,
        n:int,
        llm,
        args
)->float:
    prompts = []
    priorities = []
    sampling_params = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        priorities.append(output_len)
        # TODO: RESTRICT MAX TOKENS??
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0,
                top_p=1.0,
                ignore_eos=False,
                max_tokens=output_len,
            )
        )
    start = time.perf_counter()
    if args.scheduling_policy == 'priority' or \
        args.scheduling_policy == 'priority_round_robin':
        print('priority/rr scheduling policy')
        llm.generate(prompts, sampling_params, 
                     priority=priorities, use_tqdm=True)
    else:
        print('fcfs scheduling policy')
        llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start

# MAIN
######################################################################
def main(args):
    print(args)
    random.seed(args.seed)
    engine_args=EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    tokenizer=llm.get_tokenizer()

    requests = sample_requests(tokenizer, args)
    elapsed_time = run_vllm(requests, args.n, llm, args)
    # TODO: GET TOTAL NUM TOKENS
    total_num_tokens = 0
    total_output_tokens = 0
    for _, prompt_len, output_len in requests:
        total_num_tokens += (prompt_len + output_len)
        total_output_tokens += total_output_tokens
    
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
    return

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    # TODO: add better description of dataset
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset. The dataset is expected to "
                        "be a json in form")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    # TODO: Use model tokenizer later
    main(args)
