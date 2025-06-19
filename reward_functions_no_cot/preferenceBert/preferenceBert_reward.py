import torch
import json, re, os, ray
from qa_metrics.RewardBert import RewardBert



def get_last_line(text: str) -> str:
    lines = text.split('\n')
    return '\n'.join(lines[-1:])

@ray.remote(num_gpus=1)
class TransformerModelActor:
    def __init__(self):
        ## Refer to https://huggingface.co/IntelligenceLab/RewardPreferenceBert for using this model
        self.model = RewardBert(device='cuda')
    
    def compute_score(self, extracted_answer, label, prompt_last_line):
        # This returns a tuple (normalized_score, final_score)
        # return get_score(self.model, self.tokenizer, label, extracted_answer)
        return self.model.compute_score(label, extracted_answer)

# Instantiate the actor once.
tm_actor = TransformerModelActor.remote()

def reward_func(queries, prompts, labels, save_path="../../completions/el5/1.5B-no-cot-mixed/el5-preferenceBert.jsonl"):
    """
    Computes scores for each example using the Ray actor.
    """
    data_entries = []  # To store data for file writing.
    futures = []       # To store remote call futures.

    # Build the list of futures and record the related data.
    for query, prompt, label in zip(queries, prompts, labels):
        # Extract response.
        response = query[len(prompt):].strip() if query.startswith(prompt) else query
        
        # Extract generated answer.
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else response
        
        # Queue the remote call.
        future = tm_actor.compute_score.remote(extracted_answer, label, get_last_line(prompt))
        futures.append(future)
        
        # Save data for later writing.
        data_entries.append({
            "Prompt": prompt,
            "Response": response,
            "ExtractedResponse": extracted_answer,
            "Label": label
        })

    # Retrieve all scores concurrently.
    # Each element in `scores` is a tuple: (normalized_score, final_score)
    scores = ray.get(futures)
    
    rewards = []
    with open(save_path, "a") as file:
        for data, (normalized_score, final_score) in zip(data_entries, scores):
            data["Reward"] = {"normalized_score": normalized_score, "final_score": final_score}
            rewards.append(normalized_score)
            file.write(json.dumps(data) + "\n")
    
    return torch.tensor(rewards)

# Example usage:
# if __name__ == "__main__":
#     queries = [
#         "Hello, how are you? <think> I am thinking... </think>\n<answer> I'm fine, thank you.</answer>",
#         "Hi there! What's the weather like? <think> Considering the forecast... </think>\n<answer> It's sunny and warm.</answer>"
#     ]
#     prompts = [
#         "Hello, how are you?",
#         "Hi there! What's the weather like?"
#     ]
#     labels = [
#         "I'm fine, thank you.",
#         "It's sunny and warm."
#     ]
    
#     reward_scores = reward_func(queries, prompts, labels)
#     print("Reward scores:", reward_scores)