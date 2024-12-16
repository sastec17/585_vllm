# Custom Benchmarking Data

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

## Adding noise
Modify the model name at the top of `data/add_noise.py`
Run the following:
```bash
$ python3 data/add_noise.py
```
This will add varying noise levels to the existing dataset, and will store the output in a fresh file at `data/<model>/<output_len>/<model>_annotated_data.json`