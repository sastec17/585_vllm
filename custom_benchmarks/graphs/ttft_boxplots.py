import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# EDITABLE
MODEL = "facebook_opt-125m"
OUTPUT_LENGTH = 512
# STOP EDITING


JSON_DIR = f"../data/{MODEL}/{OUTPUT_LENGTH}/"
JSONS = {
    "FCFS": "fcfs_o.json",
    "SJF": "priority_o.json",
    "Round-robin SJF": "priority_round_robin_o.json",
}
DATA_STRING = "ttfts"

GRAPH_TITLE = "Scheduling Algorithm TTFTs"
X_LABEL = "Scheduling Policy"
Y_LABEL = "TTFT (s)"
# adjust graph fontsizes changing stuff


def read_json_data(file_path) -> float:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[DATA_STRING]


# Read data from JSONs
data = []
for name, file in JSONS.items():
    data.extend(
        [{X_LABEL: name, Y_LABEL: ttft} for ttft in read_json_data(JSON_DIR + file)]
    )


df = pd.DataFrame(data)

sns.boxplot(df, x=X_LABEL, y=Y_LABEL)

plt.title(GRAPH_TITLE)
plt.savefig(f"ttft_boxplots_{MODEL}_{OUTPUT_LENGTH}.png")
