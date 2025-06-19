import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge   # Using Ridge (a linear regression with L2 regularization)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datasets import load_dataset, load_from_disk, concatenate_datasets, Value
from rouge_score import rouge_scorer
from tqdm import tqdm
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion

# ---------------------------
# Custom Transformers
# ---------------------------

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column from the DataFrame."""
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

class RougeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L f-measure scores for each sample.
    Expects columns: 'orig_reference_answer' and 'orig_response'.
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
        # Return an array of shape (n_samples, 3)
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

# ---------------------------
# Custom Regressor Wrapper
# ---------------------------
class BoundedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps a Ridge regressor and clips the predictions to always be in [0, 1].
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.base_regressor = Ridge(alpha=self.alpha)
        
    def fit(self, X, y):
        self.base_regressor.fit(X, y)
        return self
    
    def predict(self, X):
        preds = self.base_regressor.predict(X)
        # Clip predictions to ensure outputs are within [0, 1]
        return np.clip(preds, 0, 1)

class NormalizedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps a Ridge regressor and applies a sigmoid transformation to its predictions.
    This ensures that the output is always in the range (0, 1).
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.base_regressor = Ridge(alpha=self.alpha)
        
    def fit(self, X, y):
        self.base_regressor.fit(X, y)
        return self
    
    def predict(self, X):
        preds = self.base_regressor.predict(X)
        # Apply the sigmoid (logistic) transformation for normalization.
        normalized_preds = 1 / (1 + np.exp(-preds))
        return normalized_preds

# ---------------------------
# Build the Model Pipeline
# ---------------------------
# Feature branch for TF-IDF features on the combined text.
tfidf_pipeline = Pipeline([
    ('selector', ColumnSelector("combined_text")),
    ('tfidf', TfidfVectorizer())
])

# Feature branch for ROUGE-based features.
rouge_pipeline = Pipeline([
    ('rouge', RougeFeatureTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
])

# Feature branch for the length penalty feature.
length_pipeline = Pipeline([
    ('length', LengthPenaltyTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
])

combined_features = FeatureUnion([
    ('tfidf', Pipeline([
        ('selector', ColumnSelector("combined_text")),
        ('tfidf', TfidfVectorizer())
    ])),
    ('rouge', Pipeline([
        ('rouge', RougeFeatureTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
    ])),
    ('length', Pipeline([
        ('length', LengthPenaltyTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
    ]))
])

# Create the final pipeline using NormalizedRidgeRegressor.
model_pipeline = Pipeline([
    ('features', combined_features),
    ('regressor', NormalizedRidgeRegressor(alpha=1.0))
])

def load_model(model_path='./lr_checkpoints/model_pipeline.pkl'):
    """
    Load the saved model pipeline from disk.

    Parameters:
        model_path (str): Path to the saved model pipeline file (default "model_pipeline.pkl").

    Returns:
        model_pipeline: The loaded model pipeline.
    """
    try:
        model_pipeline = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model_pipeline
    except Exception as e:
        raise ValueError(f"Error loading the model from {model_path}: {e}")

def predict_quality(model_pipeline, reference_answer, candidate_answer):
    """
    Predict the quality score for a candidate answer given the reference answer.
    
    This function builds the input DataFrame for the model pipeline, gets the predicted
    quality score (which is normalized via a sigmoid), and then applies a length similarity
    penalty. The penalty is computed as:
    
        length_penalty = min(n_ref, n_candidate) / max(n_ref, n_candidate)
    
    The final score is the product of the predicted quality and the length_penalty.
    
    Parameters:
        model_pipeline: The trained model pipeline.
        reference_answer (str): The reference answer text.
        candidate_answer (str): The candidate answer text.
        
    Returns:
        float: The final quality score in the range [0, 1].
    """
    # Combine the reference and candidate answer with a [SEP] token.
    combined_text = f"{reference_answer} [SEP] {candidate_answer}"
    
    # Build the input DataFrame for the pipeline.
    input_data = pd.DataFrame({
        "combined_text": [combined_text],
        "orig_reference_answer": [reference_answer],
        "orig_response": [candidate_answer]
    })
    
    # Get the predicted quality score (already normalized via sigmoid in the regressor).
    predicted_score = model_pipeline.predict(input_data)[0]
    
    # Compute the length similarity penalty.
    # This penalty is 1.0 if the candidate answer length is identical to the reference,
    # and lower when there is a mismatch.
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

# Example usage:
if __name__ == "__main__":
    # ===========================
    # Data Preparation
    # ===========================
    # Load the two datasets.
    raw_dataset = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    raw_dataset1 = load_from_disk('/fs/nexus-scratch/zli12321/active-topic-modeling/reward_model_tune/mocha_processed')
    
    # Cast the "orig_score" column to float.
    raw_dataset = raw_dataset.cast_column("orig_score", Value("float64"))
    raw_dataset1 = raw_dataset1.cast_column("orig_score", Value("float64"))
    
    print("Total samples in raw_dataset:", len(raw_dataset))
    
    # Split raw_dataset into train/val/test splits.
    train_val_test = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_data = train_val_test["train"]
    test_temp = train_val_test["test"]
    val_test = test_temp.train_test_split(test_size=0.5, seed=42)
    val_data = val_test["train"]
    test_data = val_test["test"]
    
    # For demonstration, we print the datasets and then concatenate train_data with raw_dataset1.
    print(raw_dataset)
    print(raw_dataset1)
    train_data = concatenate_datasets([train_data, raw_dataset1])
    
    # Convert the Hugging Face dataset to a Pandas DataFrame.
    df_train = train_data.to_pandas()
    
    # Create a new column "combined_text" that concatenates the reference answer and candidate response,
    # with a "[SEP]" token in between.
    df_train["combined_text"] = df_train["orig_reference_answer"] + " [SEP] " + df_train["orig_response"]
    
    # Normalize the original scores (originally in the range [1, 5]) to the range [0, 1].
    df_train["quality_score"] = (df_train["orig_score"] - 1.0) / 4.0
    
    # Define the features and target.
    X = df_train[["combined_text", "orig_reference_answer", "orig_response"]]
    y = df_train["quality_score"]
    
    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ===========================
    # Build the Pipeline with Updated Architecture
    # ===========================
    # Pipeline for TF-IDF features on the combined text.
    tfidf_pipeline = Pipeline([
        ('selector', ColumnSelector("combined_text")),
        ('tfidf', TfidfVectorizer())
    ])
    
    # Pipeline for ROUGE-based features.
    rouge_pipeline = Pipeline([
        ('rouge', RougeFeatureTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
    ])
    
    # Pipeline for the length penalty feature.
    length_pipeline = Pipeline([
        ('length', LengthPenaltyTransformer(reference_col='orig_reference_answer', candidate_col='orig_response'))
    ])
    
    # Combine the three feature branches.
    combined_features = FeatureUnion([
        ('tfidf', tfidf_pipeline),
        ('rouge', rouge_pipeline),
        ('length', length_pipeline)
    ])
    
    # Final pipeline: transform input with the feature union and then regress using NormalizedRidgeRegressor.
    model_pipeline = Pipeline([
        ('features', combined_features),
        ('regressor', NormalizedRidgeRegressor(alpha=1.0))
    ])
    
    # ===========================
    # Train the Model
    # ===========================
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    
    # ===========================
    # Evaluation
    # ===========================
    y_pred = model_pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("Validation MSE:", mse)
    print("Validation R2 Score:", r2)
    
    # Save the trained model.
    model_save_path = "./lr_checkpoints/model_pipeline.pkl"
    joblib.dump(model_pipeline, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    # ===========================
    # Prediction Example
    # ===========================
    # For prediction, we re-load the model using joblib.load.
    def load_model(model_path):
        try:
            model_pipeline = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model_pipeline
        except Exception as e:
            raise ValueError(f"Error loading the model from {model_path}: {e}")
    
    def predict_quality(model_pipeline, reference_answer, candidate_answer):
        # Combine reference and candidate with a "[SEP]" token.
        combined_text = f"{reference_answer} [SEP] {candidate_answer}"
        input_data = pd.DataFrame({
            "combined_text": [combined_text],
            "orig_reference_answer": [reference_answer],
            "orig_response": [candidate_answer]
        })
        predicted_score = model_pipeline.predict(input_data)
        return predicted_score[0]
    
    # Load the model.
    model = load_model(model_save_path)
    
    # Example reference and candidate answers.
    reference = "The capital of France is Paris."
    candidate = "Paris is the capital city of France."
    
    # Predict the quality score.
    quality_score = predict_quality(model, reference, candidate)
    print("Predicted Quality Score (normalized via sigmoid):", quality_score)