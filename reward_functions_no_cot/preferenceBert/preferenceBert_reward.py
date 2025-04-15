import torch
import torch.nn as nn
import json, re, os, ray
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import save_file, load_file
from datasets import Value

# Define the model
class BertAnswerScorer(nn.Module):
    """
    A simple BERT-based regressor to predict a single score in [0, 1].
    """
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None and getattr(self.bert.config, "type_vocab_size", 1) > 1:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        # Use pooler_output if available; otherwise, take the CLS token representation.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        score = torch.sigmoid(logits).squeeze(-1)
        return score

def load_model_and_tokenizer_from_folder(checkpoint_folder, base_model_name="answerdotai/ModernBERT-base", device=None):
    """
    Loads the model and tokenizer from the given checkpoint folder.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    config = AutoConfig.from_pretrained(checkpoint_folder)
    
    model = BertAnswerScorer(base_model_name)
    
    safe_path = os.path.join(checkpoint_folder, "pytorch_model.safetensors")
    state_dict = load_file(safe_path)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def get_score(model, tokenizer, reference, generated_response, device=None):
    MAX_LENGTH = 2048
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    combined_text = f"{reference} [SEP] {generated_response}"
    
    encoding = tokenizer(
        combined_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    
    model.eval()
    with torch.no_grad():
        normalized_score = model(**encoding)
    
    # Map normalized score to the [1, 5] scale.
    final_score = 1.0 + 4.0 * normalized_score.item()
    return normalized_score.item(), final_score

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    return '\n'.join(lines[-1:])

@ray.remote(num_gpus=1)
class TransformerModelActor:
    def __init__(self):
        checkpoint_folder = os.path.join(
            "/fs/nexus-scratch/zli12321/active-topic-modeling/reward_model_tune/modernbert_checkpoints",
            "checkpoint-epoch-3"
        )
        self.model, self.tokenizer = load_model_and_tokenizer_from_folder(
            checkpoint_folder, base_model_name="answerdotai/ModernBERT-base"
        )
    
    def compute_score(self, extracted_answer, label, prompt_last_line):
        # This returns a tuple (normalized_score, final_score)
        return get_score(self.model, self.tokenizer, label, extracted_answer)

# Instantiate the actor once.
tm_actor = TransformerModelActor.remote()

def reward_func(queries, prompts, labels, save_path="/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/el5/1.5B-no-cot-mixed/el5-preferenceBert.jsonl"):
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