import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# EDITABLE
MODEL = "meta-llama_Llama-3.2-1B"
OUTPUT_LENGTH = 1024
# STOP EDITING


JSON_DIR = f"../data/{MODEL}/{OUTPUT_LENGTH}/5/"
JSONS = {
    "SJF": "priority_o.json",
    "Round Robin SJF": "priority_round_robin_o.json",
    "Round Robin": "round_robin_o.json",
    "FCFS": "fcfs_o.json",
}
DATA_STRING = "e2els"

GRAPH_TITLE = f"Scheduling Algorithm {DATA_STRING.upper()}"
X_LABEL = "Scheduling Policy"
Y_LABEL = f"{DATA_STRING.upper()} (s)"
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


sns.violinplot(df, x=X_LABEL, y=Y_LABEL)
plt.title(GRAPH_TITLE)
plt.savefig(f"{DATA_STRING}_violinplots_{MODEL}_{OUTPUT_LENGTH}.png")

sns.boxplot(df, x=X_LABEL, y=Y_LABEL)
plt.title(GRAPH_TITLE)
plt.savefig(f"{DATA_STRING}_boxplots_{MODEL}_{OUTPUT_LENGTH}.png")
