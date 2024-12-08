import json

# File paths for the two JSON files
file1 = "meta-llama_Llama-3.2-1B_data_1.json"
file2 = "meta-llama_Llama-3.2-1B_data_2.json"
output_file = "meta-llama_Llama-3.2-1B_data.json"

# Load the content of the two JSON files
with open(file1, 'r') as f1, open(file2, 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Combine the two lists
combined_data = data1 + data2

# Write the combined data to a new JSON file
with open(output_file, 'w') as f_out:
    json.dump(combined_data, f_out, indent=4)

print(f"Combined JSON file saved to {output_file}")
