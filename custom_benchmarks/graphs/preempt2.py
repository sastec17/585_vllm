import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# EDITABLE
MODEL = "facebook_opt-125m"
OUTPUT_LENGTH = 512
# STOP EDITING


JSON_DIR = f"../data/{MODEL}/test_preempt/"
JSONS = {x: f"preempt_{x}_o.json" for x in [25, 50, 60, 75, 100, 125, 150, 175, 200]}
DATA_STRINGS = ["mean_e2el_ms", "ttfts"]

GRAPH_TITLE = "Round Robin SJF E2EL"
X_LABEL = "Tokens/Pre-emption"
Y_LABEL_E2EL = "E2EL (ms)"
Y_LABEL_TTFT = "TTFT (s)"
# adjust graph fontsizes changing stuff


def read_json_data(file_path) -> float:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[DATA_STRINGS[0]], data[DATA_STRINGS[1]]


# e2el graphs
data = []
for name, file in JSONS.items():
    data.append({X_LABEL: name, Y_LABEL_E2EL: read_json_data(JSON_DIR + file)[0]})


df = pd.DataFrame(data)

sns.lineplot(df, x=X_LABEL, y=Y_LABEL_E2EL)
plt.title(GRAPH_TITLE)

plt.savefig(f"preempt/e2el_{MODEL}_{OUTPUT_LENGTH}.png")


# ttfts
data = []
for name, file in JSONS.items():
    data.extend(
        [
            {X_LABEL: name, Y_LABEL_TTFT: ttft}
            for ttft in read_json_data(JSON_DIR + file)[1]
        ]
    )


df = pd.DataFrame(data)

sns.boxplot(df, x=X_LABEL, y=Y_LABEL_TTFT)
plt.title("Round Robin SJF TTFT")
plt.savefig(f"preempt/ttft_{MODEL}_{OUTPUT_LENGTH}.png")
