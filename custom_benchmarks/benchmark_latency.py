"""Benchmark the latency of processing a single batch of requests.

Use Oracle w/ output lengths to establish priority

Example usage:
python3 custom_benchmarks/benchmark_latency.py \
    --input-json data/gpt2_output.json \
    --model gpt2 \
    --scheduling-policy priority_round_robin \
    --output-json data/gpt2_rr_latency.json

Inspired by benchmarks/benchmark_latency.py
"""
import argparse
import dataclasses
import json
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from transformers import PreTrainedTokenizerBase

# HELPER FUNCTIONS 
######################################################################
def _get_data(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase
):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
        dataset = dataset[0:100]
    # Shuffle dataset
    random.shuffle(dataset)
    filtered_dataset: List[Tuple[str, int]] = []
    for i in range(len(dataset)):
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
        filtered_dataset.append((prompt, output_len))
    return filtered_dataset


# MAIN
######################################################################
def main(args: argparse.Namespace):
    random.seed(args.seed)
    engine_args = EngineArgs.from_cli_args(args)

    llm = LLM(**dataclasses.asdict(engine_args))
    tokenizer = llm.get_tokenizer()
    isinstance(tokenizer, PreTrainedTokenizerBase)
    filtered_dataset = _get_data(dataset_path=args.input_json,
                                 num_requests=args.num_iters,
                                 tokenizer=tokenizer)
   
    # Separate prompts and priorities
    prompts = []
    sampling_params = []
    priorities = []
    for prompt, output_token_len in filtered_dataset:
        prompts.append(prompt)
        priorities.append(output_token_len)
        sampling_params.append(
            SamplingParams(
                    n=args.n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=False,
                    max_tokens=output_token_len,
            )
        )
    # Write two versions of data extraction function 
    # Use random CL flag with model -> extract data we need + initialize LLM as needed 
    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(prompts,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            if args.scheduling_policy == 'priority' or \
                args.scheduling_policy=='priority_round_robin':
                llm.generate(prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                            priority=priorities)

            # fcfs scheduling policy
            else:
                llm.generate(prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False)

            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency
    
    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    # TODO: ADD IN PROFILE STUFF FROM OG FILE
    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')
    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    # parse CL arguments
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    # parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        '--input-json',
        type=str,
        default=None,
        required=True,
        help='Path to input data file.'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)