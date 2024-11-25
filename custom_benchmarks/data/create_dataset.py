"""Construct dataset from chatbot_arena conversations prompts.

PREREQUISITE: Run data/download_queries.py to download chatbot_arena 
queries first

Example usage:
python3 data/create_dataset.py \
    --model gpt2 \
    --output-json data/output.json
"""
import dataclasses
import json
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    # initialize LLM instance with engineArgs (see benchmarking stuff)
    llm = LLM(**dataclasses.asdict(engine_args))

    # naturally stop when output token seen
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        top_p=1.0,
        ignore_eos=False,
        max_tokens=args.output_len,
    )
    output_arr = []
    # Open chatbot_arena dataset
    print('opening dataset')
    # TODO: MAKE SURE THIS WORKS
    tokenizer=llm.get_tokenizer()
    with open('data/chatbot_queries.json', 'r') as file:
        data = json.load(file)
        data = data[0:2000]
        # TODO: Limit data for initial testing purposes
        # process in batches
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i+args.batch_size]
            # TODO: tokenize batch?
            outputs = llm.generate(batch, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                # get number of tokens
                output_tokens = tokenizer.encode(generated_text)

                # IF output argument flag on, append to return json
                output_arr.append(
                    {'input': prompt,
                     'output_len': len(generated_text),
                     'output_tokens': len(output_tokens)})

        with open(f"data/{args.model}_data.json", 'w') as json_file:
            json.dump(output_arr, json_file, indent=4)
    return

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Create custom dataset for model.")
    # Append output? Will be false otherwise Add as CL argument
    # CL argument for binned vs. exact - Default to exact
    parser.add_argument('--batch-size', type=int, default=32)
    # TODO: CHANGE DEFAULT OUTPUT SIZE? 
    parser.add_argument('--output-len', type=int, default=512)
    parser.add_argument(
        '--append-output',
        type=bool,
        default=False,
        help='Can choose to append output for each query to outputted JSON')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
