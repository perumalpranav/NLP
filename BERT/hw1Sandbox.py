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


# --- Dataset Class (mirrors hw2Sandbox) ---
class SarcasmDataset(Dataset):
    def __init__(self, data: List[Dict], featurizer, feature_mode):
        """
        Args:
            data: List of dictionaries (from get_data)
            featurizer: Instance of TextFeaturizer
            feature_mode: Callable (e.g. featurizer.to_word2vec) used to
                          convert a headline string into a feature vector
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
    dataloader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> Tuple[List[float], List[float]]:
    """
    Returns: loss_history, acc_history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    loss_history: List[float] = []
    acc_history: List[float] = []

    for e in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device).squeeze()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward pass & optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate accuracy stats
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total

        loss_history.append(avg_loss)
        acc_history.append(accuracy)

        print(f"Epoch {e + 1}/{epochs} - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    return loss_history, acc_history


if __name__ == "__main__":
    print("--- Starting Assignment Execution ---")

    # Fixed seed for reproducibility (consistent with hw2Sandbox)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device (consistent with hw2Sandbox)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        print("Loading data...")
        train_data = get_data('train.jsonl')
        valid_data = get_data('valid.jsonl')

        train_corpus = [str(d['headline']) for d in train_data]
        print(f"Loaded {len(train_data)} training samples.")
    except NotImplementedError:
        print("Error: You must implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Setup Featurizer
    try:
        # Note: DO NOT change the w2v_path. The .bin file must be placed at the top-level directory.
        featurizer = TextFeaturizer(train_corpus, w2v_path="GoogleNews-vectors-negative300.bin")

        # --- SELECT FEATURE MODE HERE ---
        feature_mode = featurizer.to_word2vec  # Change to to_one_hot or to_bow as needed

    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 3. Build Datasets & DataLoaders (mirrors hw2Sandbox)
    try:
        train_dataset = SarcasmDataset(train_data, featurizer, feature_mode)
        valid_dataset = SarcasmDataset(valid_data, featurizer, feature_mode)

        batch_size = 8  # Change batch size if needed

        # DO NOT CHANGE THE FOLLOWING LINES
        if int(os.environ.get("GS_TESTING_BATCH_SIZE", "0")) > 0:
            batch_size = int(os.environ["GS_TESTING_BATCH_SIZE"])
        # END OF DO NOT CHANGE

        # generator with fixed seed ensures consistent shuffling across runs
        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        exit(1)

    # 4. Initialize Model
    # Determine input_dim from a single sample
    sample_features = train_dataset[0]["features"]
    input_dim = sample_features.shape[0]

    hidden_dims: List[int] = [128, 64]  # TODO: Define hidden layer sizes (for example [128, 64] meaning two hidden layers with 128 and 64 units)
    output_dim: int = 2  # TODO: Define the dimension of the output layer (number of classes)

    try:
        model = SarcasmMLP(input_dim, hidden_dims, output_dim)
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 5. Train
    print("\n--- Training Start ---")
    learning_rate: float = 0.001  # TODO: Define learning rate
    num_epochs: int = 50  # TODO: Define number of epochs

    try:
        losses, accs = train_loop(model, train_loader, device, lr=learning_rate, epochs=num_epochs)
        print(f"Final Training Loss: {losses[-1]:.4f}")
        print(f"Final Training Accuracy: {accs[-1] * 100:.2f}%")
    except NotImplementedError as e:
        print(f"\nError: {e}")
        exit(1)

    # 6. Prediction (DO NOT MODIFY structure — mirrors hw2Sandbox)
    print("\n--- Generating Predictions ---")
    try:
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in valid_loader:
                features = batch["features"].to(device)
                logits = model(features)
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        predictions = np.array(predictions)

        output_path = "prediction.jsonl"
        with open(output_path, "w") as f:
            for i, pred in enumerate(predictions):
                record = {
                    "headline": valid_data[i]['headline'],
                    "prediction": int(pred)
                }
                f.write(json.dumps(record) + "\n")

        true_valid = np.array([d['is_sarcastic'] for d in valid_data], dtype=int)
        true_accuracy = (true_valid == predictions).sum() / len(true_valid)
        print(f"Test Accuracy: {true_accuracy}")
    except Exception as e:
        print(f"Error during prediction generation: {e}")

    # 7. Graphs
    print("\n--- Creating Graphs ---")
    epoch_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_range, losses, marker='o', color='blue', label='Training Loss')
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