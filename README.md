# Round-Robin Shortest Job First Inference Scheduling for LLMs

See original vLLM README [here](https://github.com/sastec17/585_vllm/blob/main/vLLM_README.md)

## Requirements
* OS: Linux
* Python 3.9-3.12
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
* CUDA 12.1 or higher (Our project used CUDA 12.4)

## Installation
See [vLLM docs](https://docs.vllm.ai/en/latest/getting_started/installation.html) for more info

Install vLLM as follows:
```bash
# (Recommended) Create a new conda environment.
$ conda create -n myenv python=3.12 -y
$ conda activate myenv

# Install vLLM with CUDA 12.1.
$ pip install vllm
```

Download our source code and link `vllm` Python import to modified source code:
```bash
$ git clone https://github.com/sastec17/585_vllm.git
$ cd 585_vllm
$ python3 python_only_dev.py
```

## Metrics
All required data required to run scripts already exist in this repository. More information on dataset creation can be found [here](https://github.com/sastec17/585_vllm/blob/main/custom_benchmarks/README.md)


All metrics use an online inference environment, where the chosen model is served on vLLM and a client server sends prompt requests via API call. 

With the exception of the the [Varying Request Rates](#varying-request-rates) section, all metrics send 1000 requests sequentially, without pause, to the server. This mimics an offline inference environment, but allows us to gather metrics on TTFT and E2EL at the request granularity. 

In our scripts and vLLM modifications, here is how we reference each algorithm:
* FCFS = fcfs in vLLM
* SJF = priority in vLLM
* RR = round_roubin in vLLM
* RR-SJF = priority_round_robin_reverse in vLLM

Note that you can replace our default model (meta-llama/Llama-3.2-1B) with any of vLLM's [supported models](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html)

### Tokens Generated Before Preemption
The `custom_benchmarks/test_preempt_val.sh` script will run our custom RR-SJF algorithm with the following defaults:
* model="meta-llama/Llama-3.2-1B"
* preeempt_tokens=[5,10,15,...,195,200]
* output_len=1024

Run the following within your conda env:
```bash
$ ./custom_benchmarks/test_preempt_val.sh 
```

The above defaults can be changed via the following command line flags:
* --model, -m: Set equal to desired model name
* --token, -t: Sepcify desired preempt token value. Can specify more than once (i.e. -t 10 -t 25)
* --output-len, -l: Set equal to output length restriction

Example usage with command line flags set:
```
./custom_benchmarks/test_preempt_val.sh --model "meta-llama/Llama-3.2-1B" -t 5 -t 10 --output-len 1024
```

### Scheduling Algorithm Comparison
The `custom_benchmarks/run_tests.sh` script will by default run all four algorithms (FCFS, SJF, RR-SJF, RR) with the following defaults:
* num-preempt=5
* model="meta-llama/Llama-3.2-1B"
* output-len=1024

Note that SJF is called "priority" within our script/vllm.

Run the following within your conda env:
```bash
$ ./custom_benchmarks/run_tests.sh 
```
The above defaults can be changed via the following command line flags:
* --model, -m: Set equal to desired model name
* --num-preempt, -n: Sepcify desired preempt token value
* --output-len, -l: Set equal to output length restriction
* --policy, -p: Specify scheduling policy to run inference with. Can specify more than once. 

Example usage with command line flags set:
```
./custom_benchmarks/run_tests.sh --model gpt2 --num-preempt 25 --output-len 1024 --policy fcfs --policy priority
```
### Testing Noise
The `custom_benchmarks/run_annotated.sh` script will run SJF and RR-SJF with noisy data with the following defaults:
* model="meta-llama/Llama-3.2-1B"
* num-preempt=5
* noises=(0 25 50 75 100 125 150)
* output-len=1024

Run the following within your conda env:
```bash
$ ./custom_benchmarks/run_annotated.sh 
```
The above defaults can be changed via the following command line flags:
* --model, -m: Set equal to desired model name
* --num-preempt, -n: Sepcify desired preempt token value
* --output-len, -l: Set equal to output length restriction
* --noise, -n: Specify desired noise value. Can specify more than once. Noises must match noise entries in `custom_benchmarks/data/<model>/<output_len>/<model>_annotated_data.json`

Example usage with command line flags set:
```
./custom_benchmarks/run_annotated.sh --model gpt2 --num-preempt 25 --output-len 1024 --noise 50 --noise 25
```
### Varying Request Rates
The `custom_benchmarks/test_request_rate.sh` runs FCFS, SJF, and RR-SJF scheduling algorithms with the following defaults:
* model="meta-llama/Llama-3.2-1B"
* num-preempt=5
* request_rates=(10 25 50 75 100 125 150)
* output-len=1024

Run the following within your conda env:
```bash
$ ./custom_benchmarks/test_request_rate.sh 
```
The above defaults can be changed via the following command line flags:
* --model, -m: Set equal to desired model name
* --num-preempt, -n: Sepcify desired preempt token value
* --output-len, -l: Set equal to output length restriction
* --rate, -r: Specify desired request rate per second. Can specify more than once.

Example usage with command line flags set:
```
./custom_benchmarks/test_request_rate.sh --model gpt2 --num-preempt 25 --output-len 1024 --rate 25 --rate 15
```