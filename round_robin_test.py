import dataclasses
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm import LLM, SamplingParams
import logging
import time

# Set logging level to suppress INFO and DEBUG messages.
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


engine_args = EngineArgs(model="facebook/opt-125m", scheduling_policy="priority_round_robin", preemption_mode="swap")

# Sample prompts.
prompts = [
    f"Sample prompt {i}" for i in range(1000)  # Generate a large number of prompts.
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(**dataclasses.asdict(engine_args))

start_time = time.time()

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
