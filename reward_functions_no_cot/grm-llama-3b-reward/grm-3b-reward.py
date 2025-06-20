import torch, json, re, ray, math
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create the large object and put it in Ray's object store.
def normalize_sigmoid(r: float) -> float:
    """Map any real r to (0,1) with a sigmoid."""
    return 1.0 / (1.0 + math.exp(-r))

@ray.remote(num_gpus=1)
class TransformerModelActor:
    def __init__(self):
        self.device = 'cuda'
        self.tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                        'Ray2333/GRM-Llama3.2-3B-rewardmodel-ft', torch_dtype=torch.float16, 
                        device_map=self.device,
                        )
        self.kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
    
    def compute_score(self, message, response):
        message = [
        {'role': 'user', 'content': message},
        {'role': 'assistant', 'content': response}
        ]

        message_template = self.tokenizer.apply_chat_template(message, tokenize=False)
        tokens = self.tokenizer.encode_plus(message_template, **self.kwargs)

        with torch.no_grad():
            reward_tensor = self.reward_model(tokens["input_ids"][0].view(1,-1).to(self.device), attention_mask=tokens["attention_mask"][0].view(1,-1).to(self.device))[0]
            reward = reward_tensor.cpu().detach().item()

        return normalize_sigmoid(reward)
    
# Instantiate the actor once
tm_actor = TransformerModelActor.remote()

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    return last_line

def reward_func(queries, prompts, labels, save_path="../../completions/el5/3B-no-cot-mixed/el5-grm-3b-sigmoid.jsonl"):
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
            pedant_score = ray.get(tm_actor.compute_score.remote(prompt, response))
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
