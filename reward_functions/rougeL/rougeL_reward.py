import torch, json, re
from rouge_score import rouge_scorer

def reward_func(queries, prompts, labels, save_path="/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/el5/el5-rougeL.jsonl"):
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
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
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
            rouge_l_score = scorer.score(label, extracted_answer)['rougeL'].fmeasure
            rewards.append(rouge_l_score)

            # Save the data as a JSON object in the JSONL file.
            data_dict = {
                "Prompt": prompt,
                "Response": response,
                "ExtractedResponse": extracted_answer,
                "Reward": rouge_l_score,
                "Label": label
            }
            file.write(json.dumps(data_dict) + "\n")
    
    return torch.tensor(rewards)


# # Example usage:
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