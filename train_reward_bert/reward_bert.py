import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from datasets import Value

# Disable tokenizers parallelism and torch dynamo FX tracing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.disable()

MAX_LENGTH = 2048
BATCH_SIZE = 4
CHECKPOINT_DIR = "modernbert_checkpoints"
EPOCH=10
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

############################################
# 1) Setup for Distributed Training
############################################
def setup_distributed():
    # This will use environment variables provided by torchrun:
    # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, etc.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

############################################
# 2) Custom PyTorch Dataset
############################################
class AnswerQualityDataset(Dataset):
    """
    Stores (reference, candidate, score) and handles tokenization.
    The score is in [1..5], which we'll normalize to [0..1].
    """
    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        reference = item["orig_reference_answer"]
        candidate = item["orig_response"]
        original_score = item["orig_score"]

        # Convert score to float
        try:
            original_score = float(original_score)
        except Exception as e:
            raise ValueError(f"Failed to convert original_score '{original_score}' to float: {e}")

        # Normalize label: 1 -> 0.0, 5 -> 1.0
        label = (original_score - 1.0) / 4.0

        # Concatenate reference and candidate with a [SEP] token
        combined_text = f"{reference} [SEP] {candidate}"
        encoding = self.tokenizer(
            combined_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove extra batch dimension and add label
        item_dict = {key: val.squeeze(0) for key, val in encoding.items()}
        item_dict["labels"] = torch.tensor(label, dtype=torch.float)
        return item_dict

############################################
# 3) Model Definition (BERT Regressor)
############################################
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

############################################
# 4) Helper: Compute Accuracy
############################################
def compute_accuracy(preds, labels):
    preds_rounded = torch.round(1.0 + 4.0 * preds)
    labels_rounded = torch.round(1.0 + 4.0 * labels)
    correct = (preds_rounded == labels_rounded).float()
    return correct.mean().item()

############################################
# 5) Evaluate Function (Loss + Accuracy)
############################################
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            acc = compute_accuracy(predictions, labels)
            batch_size = labels.size(0)
            total_acc += acc * batch_size
            total_count += batch_size

    mean_loss = total_loss / len(dataloader)
    mean_acc = total_acc / total_count if total_count > 0 else 0.0
    return mean_loss, mean_acc


def save_checkpoint_full(epoch, model, tokenizer, checkpoint_dir, base_model_name="answerdotai/ModernBERT-base"):
    """
    Saves a full checkpoint for the given epoch.
    
    This creates a folder named "checkpoint-epoch-{epoch}" inside checkpoint_dir.
    Inside the folder, it saves:
      - The tokenizer (via tokenizer.save_pretrained)
      - The model configuration (via model.bert.config.save_pretrained)
      - The model weights as safetensors ("pytorch_model.safetensors")
    
    Args:
        epoch (int): The current epoch number.
        model (nn.Module): Your model (possibly wrapped with DDP).
        tokenizer: The Hugging Face tokenizer.
        checkpoint_dir (str): The parent directory in which checkpoints are saved.
        base_model_name (str): The base model name used for instantiating your model (for reference).
    """
    # Create a directory for this epoch's checkpoint.
    save_dir = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the tokenizer files.
    tokenizer.save_pretrained(save_dir)
    
    # Save the model configuration.
    # (Assumes your model has a .bert attribute with a config.)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.bert.config.save_pretrained(save_dir)
    
    # Save the model weights in safetensors format.
    state_dict = model_to_save.state_dict()
    safe_path = os.path.join(save_dir, "pytorch_model.safetensors")
    save_file(state_dict, safe_path)
    
    print(f"Saved full checkpoint for epoch {epoch} to {save_dir}")

############################################
# 6) Main Training Routine with DDP
############################################
def main():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        device, local_rank = setup_distributed()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    raw_dataset = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    raw_dataset1 = load_from_disk('./mocha_processed')
    raw_dataset = raw_dataset.cast_column("orig_score", Value("float64"))
    raw_dataset1 = raw_dataset1.cast_column("orig_score", Value("float64"))

    print("Total samples in raw_dataset:", len(raw_dataset))

    # Split dataset into train/val/test
    train_val_test = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_data = train_val_test["train"]
    test_temp = train_val_test["test"]
    val_test = test_temp.train_test_split(test_size=0.5, seed=42)
    val_data = val_test["train"]
    test_data = val_test["test"]

    print(raw_dataset)
    print(raw_dataset1)
    train_data = concatenate_datasets([train_data, raw_dataset1])


    print("Train samples:", len(train_data))
    print("Val samples:", len(val_data))
    print("Test samples:", len(test_data))
    
    # Prepare tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_dataset = AnswerQualityDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_dataset = AnswerQualityDataset(val_data, tokenizer, max_length=MAX_LENGTH)
    test_dataset = AnswerQualityDataset(test_data, tokenizer, max_length=MAX_LENGTH)
    # exit(0)
    # Create Distributed Samplers and DataLoaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=test_sampler)

    # Initialize model, loss, optimizer
    model = BertAnswerScorer("answerdotai/ModernBERT-base").to(device)
    if distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    epochs = EPOCH
    for epoch in range(1, epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_train_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", disable=(distributed and dist.get_rank() != 0)):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            batch_acc = compute_accuracy(predictions, labels)
            batch_size = labels.size(0)
            total_train_acc += batch_acc * batch_size
            total_train_count += batch_size

        train_loss = total_train_loss / len(train_loader)
        train_acc = total_train_acc / total_train_count if total_train_count > 0 else 0.0

        # Evaluate on validation and test sets
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Only rank 0 prints and saves checkpoints
        if not distributed or dist.get_rank() == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

            # Save a full checkpoint folder for this epoch
            save_checkpoint_full(epoch, model, tokenizer, CHECKPOINT_DIR, base_model_name="answerdotai/ModernBERT-base")

    # Optionally, you could also save a final checkpoint folder if desired.
    if not distributed or dist.get_rank() == 0:
        final_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "checkpoint-final")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        tokenizer.save_pretrained(final_checkpoint_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.bert.config.save_pretrained(final_checkpoint_dir)
        state_dict = model_to_save.state_dict()
        safe_path = os.path.join(final_checkpoint_dir, "pytorch_model.safetensors")
        save_file(state_dict, safe_path)
        print(f"Saved final full model checkpoint to {final_checkpoint_dir}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
