""" 
    Download and clean chatbot_arena_conversations dataset.
    Currently only save query to chatbot_queries.json

    Commented out code for storing entire conversation for each convo
    Note this dataset stores Q&A across many different models
"""
# Requires a 'pip install datasets' to run
from datasets import load_dataset
import pandas as pd
import json

# HELPER FUNCTIONS
##########################################################

def _write_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

##########################################################
if __name__ == "__main__":
    # Login using e.g. `huggingface-cli login` on terminal to access this dataset
    ds = load_dataset('lmsys/chatbot_arena_conversations', split="train")

    df = pd.DataFrame(ds)
    df = df[['conversation_a', 'conversation_b']]

    # conversation_a = []
    queries_a = []
    for conversation in df['conversation_a']:
        curr_convo = {'q': conversation[0]['content'],   # Position 0 hold the query
                    'a': conversation[1]['content']}     # Position 1 holds the response
        # conversation_a.append(curr_convo)

        queries_a.append(conversation[0]['content'])

    # _write_json('data/chatbot_conversations.json', conversation_a)
    _write_json('data/chatbot_queries.json', queries_a)
