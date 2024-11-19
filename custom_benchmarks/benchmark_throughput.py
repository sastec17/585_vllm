"""
Benchmark the throughput of processing a fixed number of requests.

Use Oracle w/ output lengths to establish priority

Example usage:
python3 custom_benchmarks/benchmark_throughput.py \
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
from typing import List, Optional

import torch
import uvloop
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

# sample_requests()
# * Load dataset
# * Shuffle dataset
#     * Get prompt / completion
#     * Tokenize + prune + append to filtered_dataset


# # TODO: Ensure llm loaded properly for each mode
# run_vllm():
# * Llm instance instantiation 
# * Append prompts to prompt list 
#     * What up with the multi_modal_data business??
# * Append sampling_params to array 
# * Append priorities to array based on predicted output length 

# Assume no beam search 
# * Start timer
# * Generate w prompts/ sampling params
#     * Only pass in priorities if rr or priority scheduling policy

# MAIN
######################################################################
def main(args):
    return
# * Takes in args
# * Random seed 
# * Set up tokenizer 
#     - Get request 
#     - Obtain elapsed time by running vLLM 
# * Get total num tokens 

# * Again, only use vllm stuff


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    # TODO: Add arguments
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
#     * CL processing
#         * Take out backend and hf-max-batch-size 
#         * Take out async engine and disable-frontend-multiprocessing 
#     * Set tokenizer to match model
#     * Pair down to only vllm at the bottom
    main(args)
