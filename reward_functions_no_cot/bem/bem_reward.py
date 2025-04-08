import torch, json, re, ray
from qa_metrics.transformerMatcher import TransformerMatcher

# Create the large object and put it in Ray's object store.
@ray.remote(num_gpus=1)
class TransformerModelActor:
    def __init__(self):
        self.tm = TransformerMatcher("zli12321/answer_equivalence_bert")
    
    def compute_score(self, extracted_answer, label, prompt_last_line):
        return self.tm.get_score(extracted_answer, label, prompt_last_line)
    
# Instantiate the actor once
tm_actor = TransformerModelActor.remote()

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    return last_line

def reward_func(queries, prompts, labels, save_path="/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/el5/el5-bem.jsonl"):
    """
    For each example:
      1. Extracts the response by removing the prompt from the query.
      2. Attempts to extract the generated answer from the response using the format:
             <think> ... </think>
             <answer> GENERATED_ANSWER </answer>
         If not found, uses the whole response.
      3. Saves the prompt, full response, extracted answer, and label as JSON on a separate line.
      4. Computes the pedant score (using TransformerMatcher) between the label and the extracted answer.
    
    Args:
        queries (list[str]): Each element is a concatenation of a prompt and a response.
        prompts (list[str]): The prompt portion for each example.
        labels (list[str]): The reference answers.
        tm_ref (optional): A Ray object store reference to the TransformerMatcher. If None, a global reference is used.
        save_path (str): Path to the JSONL file to save the data.
    
    Returns:
        torch.Tensor: A tensor of computed scores for each example.
    """
    # Use the global reference if no reference was passed.
    # tm = TransformerMatcher("zli12321/answer_equivalence_bert")
    rewards = []
    
    with open(save_path, "a") as file:
        for query, prompt, label in zip(queries, prompts, labels):
            # Remove the prompt from the query to isolate the response.
            if query.startswith(prompt):
                response = query[len(prompt):].strip()
            else:
                response = query  # Fallback if prompt is not a prefix.
            
            # Attempt to extract the generated answer from the response.
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if answer_match:
                extracted_answer = answer_match.group(1).strip()
            else:
                extracted_answer = response

            # Compute the pedant score using the large object.
            # pedant_score = tm.get_score(extracted_answer, label, get_last_line(prompt))
            pedant_score = ray.get(tm_actor.compute_score.remote(extracted_answer, label, get_last_line(prompt)))
            rewards.append(pedant_score)

            # Save the data as a JSON object in the JSONL file.
            data_dict = {
                "Prompt": prompt,
                "Response": response,
                "ExtractedResponse": extracted_answer,
                "Reward": float(pedant_score),
                "Label": label
            }
            file.write(json.dumps(data_dict) + "\n")
    
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