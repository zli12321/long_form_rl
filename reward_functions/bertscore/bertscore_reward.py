import torch, json, re
from bert_score import score as bert_score

def reward_func(queries, prompts, labels, 
                save_path="/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/completions/el5/el5-bertscore.jsonl"):
    """
    For each example:
      1. Extracts the response by removing the prompt from the query.
      2. Attempts to extract the generated answer from the response using the format:
             <think> ... </think>
             <answer> GENERATED_ANSWER </answer>
         If not found, uses the whole response.
      3. Saves the prompt, full response, extracted answer, computed BERTScore F1 (as Reward), and label as JSON on a separate line.
      4. Computes the BERTScore F1 between the label and the extracted answer in a batched manner.
    
    Args:
        queries (list[str]): Each element is a concatenation of a prompt and a response.
        prompts (list[str]): The prompt portion for each example.
        labels (list[str]): The reference answers.
        save_path (str): Path to the JSONL file to save the data.
    
    Returns:
        torch.Tensor: A tensor of BERTScore F1 scores for each example.
    """
    extracted_answers = []
    processed_data = []  # List of tuples: (prompt, full response, extracted answer, label)
    
    # Process each example: extract the response and then the answer
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
        
        extracted_answers.append(extracted_answer)
        processed_data.append((prompt, response, extracted_answer, label))
    
    # Compute BERTScore in a batched manner for all examples.
    # bert_score returns (P, R, F1) where each is a tensor of scores.
    _, _, f1_scores = bert_score(extracted_answers, labels, lang="en", verbose=False)
    
    # Write the results to a JSONL file.
    with open(save_path, "a") as file:
        for (prompt, response, extracted_answer, label), f1 in zip(processed_data, f1_scores):
            data_dict = {
                "Prompt": prompt,
                "Response": response,
                "ExtractedResponse": extracted_answer,
                "Reward": f1.item() if torch.is_tensor(f1) else f1,
                "Label": label
            }
            file.write(json.dumps(data_dict) + "\n")
    
    # Ensure f1_scores is a PyTorch tensor.
    if not torch.is_tensor(f1_scores):
        f1_scores = torch.tensor([float(s) for s in f1_scores])
    
    return f1_scores
