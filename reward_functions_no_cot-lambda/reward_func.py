import torch
import json


### Example reward function
def reward_func(queries, prompts, labels):
    # Compute rewards based on some property of labels, e.g., their length
    rewards = torch.tensor([len(label) for label in labels], dtype=torch.float)

    # Zip the queries, prompts, and labels together and write each set as a JSON object
    with open("/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/completions.jsonl", "a") as file:
        for query, prompt, label in zip(queries, prompts, labels):
            data_dict = {
                "Query": query,
                "Prompt": prompt,
                "Label": label
            }
            file.write(json.dumps(data_dict) + "\n")
    
    return rewards