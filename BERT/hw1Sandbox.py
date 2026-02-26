import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import json
import os
from collections import Counter
from typing import List, Dict, Tuple, Any, Union
import gensim.models
import matplotlib.pyplot as plt


# --- Utility: Data Loader ---
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        dictlist = [json.loads(l.strip()) for l in f]

    return dictlist

class SarcasmDataset(Dataset):
    def __init__(self, data: List[Dict], featurizer, feature_mode):
        """
        Args:
            data: List of dictionaries (from get_data)
            featurizer: Instance of TextFeaturizer
            feature_mode: Callable (e.g. featurizer.to_word2vec) used to convert a headline string into a feature vector
        """
        self.data = data
        self.featurizer = featurizer
        self.feature_mode = feature_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
            - features: (feature_dim,) FloatTensor of the encoded headline
            - label:    (1,) LongTensor containing the sarcasm label
        """
        item = self.data[index]
        feature_vec = self.feature_mode(item["headline"])
        features = torch.tensor(feature_vec, dtype=torch.float)
        label = torch.tensor([item["is_sarcastic"]], dtype=torch.long)
        return {
            "features": features,
            "label": label,
        }

class TextFeaturizer:
    def __init__(self, corpus: List[str], w2v_path: str = "GoogleNews-vectors-negative300.bin"):
        # The pre-trained Word2Vec model is available at:
        # https://app.box.com/s/tpmoeke56fimcbpcrdm4lvbphsjxzlc9

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        # 1. Build the vocabulary from the corpus
        self.build_vocab(corpus)
        
        # 2. Load Pre-trained Word2Vec
        print(f"Loading Word2Vec from {w2v_path}...")
        try:
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                w2v_path, binary=True
            )
            self.emb_dim = self.w2v_model.vector_size
            print("Word2Vec loaded successfully.")
        except FileNotFoundError:
            print(f"Unable to find {w2v_path}.\nPlease ensure the file exists, raise a ticket with course staff if needed.")
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer, you may also use nltk for tokenization
        return re.findall(r'\w+', text.lower())

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Constructs the vocabulary from the corpus.
        """
        self.word_to_idx = {"<UNK>": 0}
        self.idx_to_word = {0: "<UNK>"}
        counter = 1

        for text in corpus:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.word_to_idx:
                    self.word_to_idx[token] = counter
                    self.idx_to_word[counter] = token
                    counter += 1

    def to_one_hot(self, text: str) -> np.ndarray:
        """
        Convert text to a Binary Vector. Shape: (actual_vocab_size,)
        """
        vector = np.zeros(len(self.word_to_idx), dtype=int)
        
        tokens = self._tokenize(text)
        for t in tokens:
            idx = self.word_to_idx.get(t, self.word_to_idx["<UNK>"])
            vector[idx] = 1
        
        return vector

    def to_bow(self, text: str) -> np.ndarray:
        """
        Convert text to Count Vector. Shape: (actual_vocab_size,)
        """
        vector = np.zeros(len(self.word_to_idx), dtype=int)
        
        tokens = self._tokenize(text)
        for t in tokens:
            idx = self.word_to_idx.get(t, self.word_to_idx["<UNK>"])
            vector[idx] += 1
        
        return vector

    def to_word2vec(self, text: str) -> np.ndarray:
        """
        Convert text to a vector by Averaging the word embeddings.
        Shape: (emb_dim,)
        """
        tokens = self._tokenize(text)
        token_vectors = [self.w2v_model[t] for t in tokens if t in self.w2v_model]

        if token_vectors:
            return np.mean(token_vectors, axis=0)
        else:
            return np.zeros(self.emb_dim)


class SarcasmMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        super(SarcasmMLP, self).__init__()
        
        self.layers = nn.ModuleList()

        in_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_size,h))
            self.layers.append(nn.ReLU())
            in_size = h
        self.layers.append(nn.Linear(in_size, output_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x


def train_loop(
    model: nn.Module, 
    X: np.ndarray, 
    y: np.ndarray, 
    lr: float, 
    epochs: int
) -> Tuple[List[float], List[float]]:
    """
    Returns: loss_history, acc_history
    """
    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history: List[float] = []
    acc_history: List[float] = []
    
    # TODO: Implement Training Loop
    # 1. Forward pass
    # 2. Calculate loss
    # 3. Backward pass & Optimizer step
    # 4. Calculate Accuracy

    for e in range(epochs):
        outputs = model(x_tensor)

        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        preds = torch.argmax(outputs, dim=1)
        correct = (preds == y_tensor).sum().item()
        accuracy = correct / y_tensor.size(0)

        loss_history.append(loss.item())
        acc_history.append(accuracy)

        print(f"Epoch {e}: Accuracy = {accuracy}")

    
    return loss_history, acc_history


if __name__ == "__main__":
    print("--- Starting Assignment Execution ---")
    
    # 1. Load Data
    try:
        print("Loading data...")
        # Ensure these files exist in your directory
        train_data = get_data('train.jsonl')
        valid_data = get_data('valid.jsonl')
        
        train_corpus = [str(d['headline']) for d in train_data]
        train_labels = [int(d['is_sarcastic']) for d in train_data]
        print(f"Loaded {len(train_data)} training samples.")      
    except NotImplementedError:
        print("Error: You must implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Setup Features
    try:
        # Note: DO NOT change the w2v_path. The .bin file must be placed at the top-level directory.
        featurizer = TextFeaturizer(train_corpus, w2v_path="GoogleNews-vectors-negative300.bin")
        
        # --- SELECT FEATURE MODE HERE ---
        feature_mode = featurizer.to_word2vec  # Change this to to_one_hot or to_word2vec as needed
        
        x_train = np.array([feature_mode(text) for text in train_corpus])
        y_train = np.array(train_labels)
        
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 3. Initialize Model
    input_dim = x_train.shape[1]
    hidden_dims: List[int] = [128, 64]  # TODO: Define hidden layer sizes (for example [128, 64] meaning two hidden layers with 128 and 64 units)
    output_dim: int = 2  # TODO: Define the dimension of the output layer (number of classes)
    try:
        model = SarcasmMLP(input_dim, hidden_dims, output_dim)
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 4. Train
    print("\n--- Training Start ---")
    learning_rate: float = 0.001  # TODO: Define learning rate
    num_epochs: int = 50  # TODO: Define number of epochs
    try:
        losses, accs = train_loop(model, x_train, y_train, lr=learning_rate, epochs=num_epochs)
        print(f"Final Training Loss: {losses[-1]:.4f}")
        print(f"Final Training Accuracy: {accs[-1]*100:.2f}%")
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 5. Prediction (DO NOT MODIFY)
    print("\n--- Generating Predictions ---")
    try:
        valid_data = get_data('valid.jsonl')
        valid_corpus = [str(d['headline']) for d in valid_data]
        x_valid = np.array([feature_mode(text) for text in valid_corpus])
        x_valid_tensor = torch.FloatTensor(x_valid)
        
        model.eval()
        with torch.no_grad():
            logits = model(x_valid_tensor)
            predictions = torch.argmax(logits, dim=1).numpy()

        output_path = "prediction.jsonl"
        with open(output_path, "w") as f:
            for i, pred in enumerate(predictions):
                record = {
                    "headline": valid_data[i]['headline'],
                    "prediction": int(pred)
                }
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Error during prediction generation: {e}")

    true_valid = np.array([d['is_sarcastic'] for d in valid_data], dtype=int)
    correct = (true_valid == predictions).sum()
    true_accuracy = correct / len(true_valid)
    print(f"Test Accuracy: {true_accuracy}")

    # 6. Graphs
    print("\n--- Creating Graphs ---")
    epochs = range(num_epochs)

    plt.figure(figsize=(12,6))
    plt.plot(epochs, losses, marker='o', color='blue', label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs (Word2Vec)")

    plt.xticks(np.arange(0, 51, 1))
    plt.yticks(np.arange(0, 1.0, 0.1))

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig("training_loss_plot.png", dpi=300, bbox_inches='tight')
    print("Graph Successfully Created")

    print("\n--- Execution Complete ---")