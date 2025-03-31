import torch.nn as nn
import torch, json, re, os
# from tools import get_last_line
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import save_file, load_file


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

    This function expects that the checkpoint folder (e.g. "checkpoints/checkpoint-epoch-1")
    contains:
      - The tokenizer files (saved via tokenizer.save_pretrained)
      - The model configuration (saved via model.bert.config.save_pretrained)
      - The model weights saved in safetensors format ("pytorch_model.safetensors")
    
    Args:
        checkpoint_folder (str): The folder containing the checkpoint.
        base_model_name (str): The base model name (used to load the tokenizer).
        device (torch.device, optional): The device on which to load the model.
    
    Returns:
        model: The loaded model in evaluation mode.
        tokenizer: The loaded tokenizer.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer from the checkpoint folder.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    
    # (Optionally) load the configuration from the checkpoint folder.
    config = AutoConfig.from_pretrained(checkpoint_folder)
    
    # Instantiate your custom model using the base_model_name.
    # (You could also pass the config to your model if your model supports it.)
    model = BertAnswerScorer(base_model_name)
    
    # Load the model weights from the safetensors file.
    safe_path = os.path.join(checkpoint_folder, "pytorch_model.safetensors")
    state_dict = load_file(safe_path)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def get_score(model, tokenizer, reference, generated_response, device=None):
        MAX_LENGTH = 1024
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Combine the reference and generated response using a [SEP] token.
        combined_text = f"{reference} [SEP] {generated_response}"
        
        # Tokenize the input.
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
            # Get the normalized score (in [0,1]).
            normalized_score = model(**encoding)
        
        # Map normalized score to the [1,5] scale.
        final_score = 1.0 + 4.0 * normalized_score.item()
        return normalized_score.item(), final_score

checkpoint_folder = os.path.join("/fs/nexus-scratch/zli12321/active-topic-modeling/reward_model_tune/checkpoints", "checkpoint-epoch-4")
# Load the full model and tokenizer from the checkpoint folder
model, tokenizer = load_model_and_tokenizer_from_folder(checkpoint_folder, base_model_name="answerdotai/ModernBERT-base")


def reward_func(queries, prompts, labels, save_path="/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/el5/el5-preferenceBert.jsonl"):
    """
    For each example:
      1. Extracts the response by removing the prompt from the query.
      2. Attempts to extract the generated answer from the response using the format:
           <think> ... </think>
           <answer> GENERATED_ANSWER </answer>
         If not found, uses the whole response.
      3. Saves the prompt, full response, extracted answer, and label as JSON on a separate line.
      4. Computes the ROUGE-L F1 score between the label and the extracted answer.
    
    Args:
        queries (list[str]): Each element is a concatenation of a prompt and a response.
        prompts (list[str]): The prompt portion for each example.
        labels (list[str]): The reference answers.
        save_path (str): Path to the JSONL file to save the data.
    
    Returns:
        torch.Tensor: A tensor of ROUGE-L F1 scores for each example.
    """
    # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
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

            
            # Compute ROUGE-L F1 score between the label and the extracted answer.
            # rouge_l_score = scorer.score(label, extracted_answer)['rougeL'].fmeasure
            # pedant_score = tm.get_score(extracted_answer, label, get_last_line(prompt))
            pedant_score = get_score(model, tokenizer, label, extracted_answer)
            rewards.append(pedant_score)

            # Save the data as a JSON object in the JSONL file.
            data_dict = {
                "Prompt": prompt,
                "Response": response,
                "ExtractedResponse": extracted_answer,
                "Reward": pedant_score,
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