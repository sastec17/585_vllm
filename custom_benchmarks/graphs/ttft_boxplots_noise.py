import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# EDITABLE
MODEL = "facebook_opt-125m"
OUTPUT_LENGTH = 1024
# STOP EDITING


JSON_DIR = f"../data/{MODEL}/noise/"
DATA_STRING = "ttfts"

GRAPH_TITLE = "Scheduling Algorithm TTFTs"
X_LABEL = "STDEV"
Y_LABEL = "TTFT (s)"


def read_json_data(file_path) -> float:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[DATA_STRING]


# Read data from JSONs
data = []
for stdev in range(25, 151, 25):
    for policy in ["priority", "priority_round_robin"]:
        file = JSON_DIR + f"noise_{stdev}_{policy}_o.json"

        data.extend(
            [
                {X_LABEL: stdev, Y_LABEL: ttft, "Policy": policy}
                for ttft in read_json_data(file)
            ]
        )

for policy in ["priority", "priority_round_robin"]:
    file = f"../data/{MODEL}/{OUTPUT_LENGTH}/" + f"{policy}_o.json"

    data.extend(
        [{X_LABEL: 0, Y_LABEL: ttft, "Policy": policy} for ttft in read_json_data(file)]
    )

data
df = pd.DataFrame(data)
df

df.groupby(["STDEV", "Policy"]).describe()

sns.boxplot(df, x=X_LABEL, y=Y_LABEL, hue="Policy")

plt.title(GRAPH_TITLE)
plt.savefig(f"noise_ttft_boxplots_{MODEL}_{OUTPUT_LENGTH}.png")
