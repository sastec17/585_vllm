import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# EDITABLE
MODEL = "meta-llama_Llama-3.2-1B"
OUTPUT_LENGTH = 1024
# STOP EDITING


JSON_DIR = f"../data/{MODEL}/request_rate"
JSONS = {
    "SJF": "priority_o.json",
    "RR-SJF": "priority_round_robin_reverse_o.json",
    # "RR": "round_robin_o.json",
    "FCFS": "fcfs_o.json",
}
DATA_STRING = "e2els"

GRAPH_TITLE = f"Scheduling Algorithm {DATA_STRING.upper()}"
X_LABEL = "Request Rate (req/s)"
Y_LABEL = f"{DATA_STRING.upper()} (s)"
# adjust graph fontsizes changing stuff


def read_json_data(file_path) -> float:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[DATA_STRING]


# Read data from JSONs
data = []
for name, file in JSONS.items():
    for rate in [5, 10, 25, 50, 75, 100, 125, 150]:
        data.extend(
            [
                {X_LABEL: rate, Y_LABEL: ttft, "Policy": name}
                for ttft in read_json_data(f"{JSON_DIR}/{rate}_{file}")
            ]
        )


df = pd.DataFrame(data)

colors = sns.color_palette()[0:2] + [sns.color_palette()[3]]
sns.boxplot(df, x=X_LABEL, y=Y_LABEL, hue="Policy", palette=colors)
plt.title(GRAPH_TITLE)
plt.savefig(f"{DATA_STRING}_boxplots_{MODEL}_reqrate.png")
