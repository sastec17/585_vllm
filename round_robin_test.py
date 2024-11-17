import dataclasses
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm import LLM, SamplingParams

engine_args = EngineArgs(model="facebook/opt-125m", scheduling_policy="priority_round_robin")

print("created engine_args")
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Create an LLM.
print("about to create llm")
llm = LLM(**dataclasses.asdict(engine_args))
print("created llm")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("about to generate")
outputs = llm.generate(prompts, sampling_params)
print("done with generation")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
