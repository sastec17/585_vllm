import json
import matplotlib.pyplot as plt
import textwrap
import sys
import os
"""
Creates ttft.png which graphs mean TTFT from JSON_DIR json files
Pass in as many json files as you want in the order to be shown on the graph
Usage: python ttft.py
"""
input_values = sys.argv[1:] 

JSON_DIR = "~/585_vllm/custom_benchmarks/data/facebook_opt-125m/test_preempt/"
#JSON_DIR = "../data/facebook_opt-125m/test_preempt/"
JSONS = {value: f"preempt_{value}_o.json" for value in input_values}

METRICS=["ttft", "e2el"]
def main():
    for metric in METRICS:
        DATA_STRING = f"mean_{metric}_ms"

        GRAPH_FILE = f"graphs/preempt/{metric}.png"

        GRAPH_TITLE = f"Scheduling Algorithm {metric}s"
        Y_LABEL = f"{metric} (ms)"
        BAR_COLORS = ['skyblue', 'lightgreen', 'salmon']
        input_colors = {value: BAR_COLORS[i % len(BAR_COLORS)] for i, value in enumerate(input_values)}

        ALGORITHM_CHARS_PER_LINE = 12
        # adjust graph fontsizes changing stuff

        def read_json_data(file_path) -> float:
            file_path = os.path.expanduser(file_path)
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data[DATA_STRING]

        # Read data from JSONs
        data = {}
        for name, file in JSONS.items():
            data[name] = read_json_data(JSON_DIR + file)

        # Wrap long labels to take up two lines if needed
        wrapped_labels = [textwrap.fill(label, width=ALGORITHM_CHARS_PER_LINE) for label in data.keys()]

        # Create the bar graph with thinner bars
        plt.figure(figsize=(6, 6))
        plt.bar(
            wrapped_labels, 
            data.values(), 
            color=[input_colors[label] for label in input_values], 
            edgecolor='black', 
            width=0.5  # Set bar width to make them thinner
        )

        # Add title and axis labels with larger font sizes
        plt.title(GRAPH_TITLE, fontsize=25)
        plt.ylabel(Y_LABEL, fontsize=22)
        plt.xlabel("#tokens preempted", fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=12)

        # Save the bar graph
        plt.tight_layout()
        plt.savefig(GRAPH_FILE)

if __name__ == "__main__":
    main()
