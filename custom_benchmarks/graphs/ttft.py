import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

"""
Creates ttft.png which graphs mean TTFT from JSON_DIR json files
Pass in as many json files as you want in the order to be shown on the graph
Usage: python ttft.py
"""

JSON_DIR = "../data/facebook_opt-125m/512/"
JSONS = {
	"FCFS": "fcfs_o.json",
	"Priority": "priority_o.json",
	"Round-robin Shortest Job First": "priority_round_robin_o.json",
}
DATA_STRING = "mean_ttft_ms"

GRAPH_TITLE = "Scheduling Algorithm TTFTs"
Y_LABEL = "TTFT (ms)"

GRAPH_FILE = "ttft.png"

def read_json_data(file_path) -> float:
	with open(file_path, 'r') as file:
		data = json.load(file)
	return data[DATA_STRING]

# Read data from jsons
data = {}
for name, file in JSONS.items():
	data[name] = read_json_data(JSON_DIR + file)

# Create the bar graph
plt.figure(figsize=(6, 6))
plt.bar(data.keys(), data.values(), color='skyblue', edgecolor='black')

# Add title and axis labels
plt.title(GRAPH_TITLE, fontsize=16)
plt.ylabel(Y_LABEL, fontsize=14)

# Display the bar graph
plt.tight_layout()
plt.savefig(GRAPH_FILE)
