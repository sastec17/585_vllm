import json
import random

MODEL_NAME = "facebook_opt-125m"
PREEMPT_TOKENS = 1024


MIN_VAL = 15
MAX_VAL = PREEMPT_TOKENS + 1

FILENAME = f"{MODEL_NAME}/{PREEMPT_TOKENS}/{MODEL_NAME}_data.json"
FILENAME_WITH_NOISE = f"{MODEL_NAME}/{PREEMPT_TOKENS}/{MODEL_NAME}_annotated_data.json"
with open(FILENAME, "r") as file:
    data = json.load(file)

random.seed(585)
noise_levels = [10, 25, 50, 75, 100, 125, 150]  # stdevs for gaussian


def add_noise(value, std_dev):
    return max(0, round(random.gauss(value, std_dev)))


for entry in data:
    for level in noise_levels:
        field_name = f"output_tokens_noise_{level}"
        noise = add_noise(entry["output_tokens"], level)

        noise = min(noise, MAX_VAL)
        noise = max(MIN_VAL, noise)

        entry[field_name] = noise

# writeback
with open(FILENAME_WITH_NOISE, "w") as file:
    json.dump(data, file, indent=4)
