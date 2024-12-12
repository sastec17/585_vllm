import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
from glob import glob

os.getcwd()

# EDITABLE
MODEL = "meta-llama_Llama-3.2-1B"
OUTPUT_LENGTH = 1024
# STOP EDITING


glob("../data/meta-llama_llama-3.2-1B/noise/*")

JSON_DIR = f"../data/{MODEL}/noise/"
DATA_STRING = "ttfts"

GRAPH_TITLE = f"SJF vs RR-SJF {DATA_STRING.upper()} With Noise"
X_LABEL = "STDEV"
Y_LABEL = f"{DATA_STRING.upper()} (s)"


def read_json_data(file_path) -> float:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[DATA_STRING]


# Read data from JSONs
data = []
for stdev in range(0, 151, 25):
    for policy, ext in {
        "SJF": "priority",
        "RR-SJF": "priority_round_robin_reverse",
    }.items():
        file = JSON_DIR + f"noise_{stdev}_{ext}_o.json"

        data.extend(
            [
                {X_LABEL: stdev, Y_LABEL: ttft, "Policy": policy}
                for ttft in read_json_data(file)
            ]
        )

# for policy, ext in {
#     "SJF": "priority",
#     "RR-SJF": "priority_round_robin_reverse",
# }.items():
#     file = f"../data/{MODEL}/{OUTPUT_LENGTH}/" + f"{ext}_o.json"

#     data.extend(
#         [{X_LABEL: 0, Y_LABEL: ttft, "Policy": policy} for ttft in read_json_data(file)]
#     )

data
df = pd.DataFrame(data)
df

df.groupby(["STDEV", "Policy"]).describe()

sns.boxplot(df, x=X_LABEL, y=Y_LABEL, hue="Policy")

plt.title(GRAPH_TITLE)
plt.savefig(f"noise_{DATA_STRING}_boxplots_{MODEL}_{OUTPUT_LENGTH}.png")
