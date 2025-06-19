import torch, json, re, ray, joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge   # Using Ridge (a linear regression with L2 regularization)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from rouge_score import rouge_scorer
from tqdm import tqdm
import __main__ 

@ray.remote
class TransformerModelActor:
    # Nested custom transformers and regressor for proper unpickling.
    
    class ColumnSelector(BaseEstimator, TransformerMixin):
        """
        Selects a single column from the DataFrame.
        Useful for feeding a particular column (e.g. the combined text)
        into a vectorizer.
        """
        def __init__(self, column):
            self.column = column

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            # Return the column as a Series (or list) of texts
            return X[self.column]

    class RougeFeatureTransformer(BaseEstimator, TransformerMixin):
        """
        For every sample in a DataFrame (which must contain the reference and candidate texts),
        compute the ROUGE-1, ROUGE-2, and ROUGE-L f-measure scores.
        """
        def __init__(self, reference_col='orig_reference_answer', candidate_col='orig_response'):
            self.reference_col = reference_col
            self.candidate_col = candidate_col
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            for _, row in tqdm(X.iterrows(), total=len(X), desc="Computing ROUGE scores"):
                reference = row[self.reference_col]
                candidate = row[self.candidate_col]
                scores = self.scorer.score(reference, candidate)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            # Combine into a numpy array of shape (n_samples, 3)
            return np.array([rouge1_scores, rouge2_scores, rougeL_scores]).T

    class LengthPenaltyTransformer(BaseEstimator, TransformerMixin):
        """
        Computes a length similarity feature between the reference answer and candidate response.
        The feature is defined as:
            similarity = min(num_words(reference), num_words(candidate)) / max(num_words(reference), num_words(candidate))
        This yields a value between 0 and 1.
        """
        def __init__(self, reference_col='orig_reference_answer', candidate_col='orig_response'):
            self.reference_col = reference_col
            self.candidate_col = candidate_col

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            similarity_scores = []
            for _, row in tqdm(X.iterrows(), total=len(X), desc="Computing length penalty"):
                ref_text = row[self.reference_col]
                cand_text = row[self.candidate_col]
                ref_len = len(ref_text.split())
                cand_len = len(cand_text.split())
                if ref_len == 0 or cand_len == 0:
                    score = 0.0
                else:
                    score = min(ref_len, cand_len) / max(ref_len, cand_len)
                similarity_scores.append(score)
            return np.array(similarity_scores).reshape(-1, 1)

    class NormalizedRidgeRegressor(BaseEstimator, RegressorMixin):
        """
        Wraps a Ridge regressor and applies a sigmoid transformation to its predictions.
        The sigmoid function maps any real value to a value in (0, 1), ensuring that
        the final output is always normalized.
        """
        def __init__(self, alpha=1.0):
            self.alpha = alpha
            from sklearn.linear_model import Ridge
            self.base_regressor = Ridge(alpha=self.alpha)

        def fit(self, X, y):
            self.base_regressor.fit(X, y)
            return self

        def predict(self, X):
            preds = self.base_regressor.predict(X)
            normalized_preds = 1 / (1 + np.exp(-preds))
            return normalized_preds

    def __init__(self):
        # Register the nested classes into the __main__ globals so that
        # joblib.load finds them when unpickling the model pipeline.
        import __main__
        __main__.ColumnSelector = TransformerModelActor.ColumnSelector
        __main__.RougeFeatureTransformer = TransformerModelActor.RougeFeatureTransformer
        __main__.LengthPenaltyTransformer = TransformerModelActor.LengthPenaltyTransformer
        __main__.NormalizedRidgeRegressor = TransformerModelActor.NormalizedRidgeRegressor

        # Path to the saved model pipeline.
        model_path = './lr_checkpoints/model_pipeline.pkl'
        self.model = joblib.load(model_path)
    
    def compute_score(self, reference_answer, candidate_answer):
        """
        Combines the candidate and reference answers with a [SEP] token,
        builds the DataFrame expected by the pipeline, and returns the predicted quality score.
        """
        combined_text = f"{reference_answer} [SEP] {candidate_answer}"
        input_data = pd.DataFrame({
            "combined_text": [combined_text],
            "orig_reference_answer": [reference_answer],
            "orig_response": [candidate_answer]
        })
        predicted_score = self.model.predict(input_data)[0]

        ref_len = len(reference_answer.split())
        cand_len = len(candidate_answer.split())
        if ref_len == 0 or cand_len == 0:
            length_penalty = 0.0
        else:
            length_penalty = min(ref_len, cand_len) / max(ref_len, cand_len)
        
        # Optionally, you can further control the penalty with an exponent (gamma), e.g.,
        # final_score = predicted_score * (length_penalty ** gamma)
        # Here we use gamma = 1 for simplicity.
        final_score = predicted_score * length_penalty
    
        return final_score

tm_actor = TransformerModelActor.remote()

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    
    return last_line

def reward_func(queries, prompts, labels, save_path="../../completions/el5/3B-no-cot-mixed/el5-rougePedant-penalty.jsonl"):
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
            pedant_score = ray.get(tm_actor.compute_score.remote(extracted_answer, label))
            # pedant_score = pedant.get_score(extracted_answer, label, get_last_line(prompt))
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
