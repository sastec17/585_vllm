# Custom Benchmarking

## Data

To download and clean the chatbot_arena_conversations dataset from huggingface, run the following:
```
python3 data/download_queries.py 
```
This will store the queries in `data/chatbot_queries.json`

Note that the chatbot_arena_conversations dataset provides a diverse set of queries and the two corresponding responses to each query, each from different models. As we don't intend to use any of the models presented in this dataset, we intentionally only store the queries from this dataset. 


As we are implementing a priority round-robin scheduling algorithm, we need to obtain the actual outputs to whichever model we want to test. 
To do this, run the following:
```
python3 data/create_dataset.py --model gpt2
```
output will be stored in `data/{model}_data.json`


Note that you can replace the model with any of vLLM's [supported models](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html)

This script will obtain the response for each prompt by running the specified model on vLLM via offline inference. 

It will store data as follows:
```
[
    {
        'input': 'This is some test input query!',
        'output_len': 572,
        'output_tokens': 237
    }
]
```
Where `output_len` corresponds to the length of the output in characters, while `output_tokens` is the number of returned output tokens.

## Benchmarking 

To benchmark vLLM against our priority round-robin algorithm, we provide both online and offline benchmarking scripts

#### TODO: add shellscript to run full test suite / generate graphs

### Latency 

As demonstrated by vLLM's original benchmarking scripts (see `./benchmarks/`), latency can be measured by sending and timing the same batch of requests to the chosen model multiple times in a row. This allows us to benchmark roughly how long it takes to process a singular batch. 

As we need a script to load our custom dataset for any scheduling algorithm ('fcfs', 'priority', 'priority-round-robin') for comparison, we construct our own version of this script.  

To run our custom benchmarking script, please run the following (in the custom_benchmarking/ directory): 
```
python3 benchmark_latency.py --input-json data/gpt2_data.json \
    --model gpt2 \
    --scheduling-policy priority \
    --output-json data/gpt2_priority_latency.json
```

### Throughput
#### Offline throughput
As demonstrated by vLLM's original benchmarking scripts (see `./benchmarks/`), throughput can be measured by sending and timing a large batch of requests to the chosen model, then determining how fast those tokens were processed. 

As we need a script to load our custom dataset for any scheduling algorithm ('fcfs', 'priority', 'priority-round-robin') for comparison, we construct our own version of this script.  

To run our custom benchmarking script, please run the following (in the custom_benchmarking/ directory): 
```
python3 benchmark_throughput.py --input-json data/gpt2_data.json \
    --model gpt2 \
    --scheduling-policy priority \
    --output-json data/gpt2_priority_throughput.json
```

#### Online throughput
Online benchmarking requires us to run our desired model as a server. We can do this as follows:
```
vllm serve gpt2 --swap-space 16 --disable-log-requests --scheduling-policy fcfs
```
Here, we can replace `gpt2` with the desired model, and `fcfs` with the desired scheduling policy.

On the client side, run the following for online benchmarking (in the custom_benchmarking/ directory):
```
python3 online_benchmarking.py --backend vllm \
    --model gpt2 \
    --schedule fcfs \ # By default <schedule> is fcfs
    --dataset-path data/gpt2_data.json \
    --request-rate <request_rate> \ # By default <request_rate> is inf
    --num-prompts <num_prompts> # By default <num_prompts> is 1000
```
Results are by default stored in current directory with the following format: {backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json

## Considersations:

This part will be removed later. 
Guiding questions:
1. Should the model be completely deterministic for this? As in temperature=0 in all SamplingParams? I'm leaning yes, or at least maybe 0.5, as this will make our "oracle" predictions more accurate.

2. Feasible to construct a dataset with BERT output length predictions instead of just oracle?

3. Thoughts on binning oracle requests rather than using exact values. I think we can certainly test both. 

4. How restricted should our output tokens be? Should max tokens be 512? 1024? 

5. Right now, the dataset creation restricts output to 256 tokens (can certainly be changed). In benchmarking, requests are then limited to whatever output request token length originally occurred. I'm thinking this shouldn't be the case?