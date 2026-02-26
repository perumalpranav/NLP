import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Union
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Utility: Data Loader ---
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# --- Dataset Class ---
class SarcasmDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = 128):
        """
        Args:
            data: List of dictionaries (from get_data)
            tokenizer: Instance of BertTokenizer
            max_length: Maximum sequence length for truncation/padding
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
            - input_ids: (max_length,) Tensor of token IDs 
            - attention_mask: (max_length,) Tensor for attention masking
            - label: (1,) Tensor containing the label
        """
        item = self.data[index]
        encoded = self.tokenizer(
            item["headline"],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = torch.tensor([item['is_sarcastic']], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

# --- Model Class ---
class SarcasmBERT(nn.Module):
    def __init__(self):
        super(SarcasmBERT, self).__init__()
        
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.classify_layer = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, 2)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        logits = self.classify_layer(cls_vector)
        return logits

# --- Training Loop ---
def train_loop(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    lr: float, 
    epochs: int,
    **kwargs
) -> List[float]:
    
    # We use AdamW as the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            #Loaded Data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).squeeze()

            #Run Model
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step() 

            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Feel free to pass additional arguments (like validation dataloader) 
        # through kwargs and perform validation here.
        
    return loss_history

# --- Main Execution ---
if __name__ == "__main__":
    # Check for GPU/MPS
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        train_data = get_data('train.jsonl')
        valid_data = get_data('valid.jsonl')
    except NotImplementedError:
        print("Error: Implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Tokenizer & Dataset
    print("Initializing Tokenizer...")
    # TODO: Initialize BertTokenizer from 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
    
    try:
        train_dataset = SarcasmDataset(train_data, tokenizer)
        valid_dataset = SarcasmDataset(valid_data, tokenizer)
        
        # Create DataLoaders
        batch_size = 8  # Change batch size if needed

        # DO NOT CHANGE THE FOLLOWING LINES
        if int(os.environ.get("GS_TESTING_BATCH_SIZE", "0")) > 0:
            batch_size = int(os.environ["GS_TESTING_BATCH_SIZE"])
        # END OF DO NOT CHANGE

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        exit(1)

    # 3. Model
    try:
        model = SarcasmBERT()
    except NotImplementedError:
        print("Error: Implement SarcasmBERT first.")
        exit(1)

    # 4. Training
    # DO NOT CHANGE THE FOLLOWING LINES
    print("Starting Training...")
    # END OF DO NOT CHANGE
    try:
        # TODO: Set learning rate and number of epochs
        lr = 2e-5
        epochs = 5
        losses = train_loop(model, train_loader, device, lr, epochs)
    except NotImplementedError:
        print("Error: Implement train_loop.")
        exit(1)


    """
        YOUR ADDITIONAL CODE BELOW (DO NOT DELETE THIS COMMENT)
    """

    #5. Predicitions
    print("\n--- Generating Predictions ---")
    try:
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        predictions = np.array(predictions)

        output_path = "prediction.jsonl"
        with open(output_path, "w") as f:
            for i, pred in enumerate(predictions):
                record = {"headline": valid_data[i]['headline'], "prediction": int(pred)}
                f.write(json.dumps(record) + "\n")

        true_valid = np.array([d['is_sarcastic'] for d in valid_data], dtype=int)
        true_accuracy = (true_valid == predictions).sum() / len(true_valid)
        print(f"Test Accuracy: {true_accuracy}")
    except Exception as e:
        print(f"Error during prediction generation: {e}")

    # 6. Graphs
    print("\n--- Creating Graphs ---")
    epoch_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, losses, marker='o', color='blue', label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs (BERT)")

    plt.xticks(np.arange(0, 51, 1))
    plt.yticks(np.arange(0, 1.0, 0.1))

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sandbox_training_loss_plot.png", dpi=300, bbox_inches='tight')
    print("Graph Successfully Created")

    print("\n--- Execution Complete ---")