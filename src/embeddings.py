# NewDesign/src/embeddings.py

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import warnings

# Change relative import to direct import
import models

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# === Helper Functions (Tokenizer, Vocab, Encode, Dataset, Collate) ===
def tokenize(text: str):
    """Simple space-based tokenizer."""
    return text.lower().strip().split()

def build_vocab(texts, min_freq=1):
    """Builds a vocabulary from a list of texts."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    """Encodes a single text string into tensor of indices."""
    tokens = tokenize(text)
    return torch.tensor([vocab.get(token, vocab["<unk>"]) for token in tokens], dtype=torch.long)

class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    def __init__(self, texts, vocab):
        self.vocab = vocab
        self.encoded_texts = [encode(text, vocab) for text in texts]
    def __len__(self):
        return len(self.encoded_texts)
    def __getitem__(self, idx):
        return self.encoded_texts[idx]

def collate_fn(batch):
    """Pads sequences within a batch using the <pad> token index (0)."""
    return pad_sequence(batch, batch_first=True, padding_value=0)
# ========================================================================


# === Main Function matching combined.py usage ===

def create_and_apply_embeddings(
    df: pd.DataFrame,
    text_column: str,
    subsample_size: int = 5000,
    batch_size: int = 64,
    infer_batch_size: int = 128,
    num_epochs: int = 5,
    embed_dim: int = 50,
    hidden_dim: int = 64,
    learning_rate: float = 0.001,
    random_state: int = 42,
    checkpoint_dir: str = "checkpoints/lstm_classifier",
    output_dir: str = "results",
    device: str = None
):
    """
    Trains an LSTM TextEncoder on a subsample and applies it to the full dataset
    to generate embeddings, matching the interface used in combined.py.
    Saves model, vocab, sample embeddings, and full embeddings.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text column.
        text_column (str): The name of the column with text data.
        subsample_size (int): Number of samples to use for training the encoder.
        batch_size (int): Batch size for training.
        infer_batch_size (int): Batch size for generating embeddings (inference).
        num_epochs (int): Number of epochs to train the encoder.
        embed_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of LSTM hidden state.
        learning_rate (float): Learning rate for the Adam optimizer.
        random_state (int): Random seed for reproducibility.
        checkpoint_dir (str): Directory to save/load model checkpoints and vocabulary.
        output_dir (str): Directory to save the generated embeddings (sample and full).
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        pd.DataFrame: DataFrame with original indices and embeddings for the full dataset.
                      Returns None if an error occurs.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- LSTM Embedding Generation ---")
    print(f"Using device: {device}")

    # --- Setup Paths ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "text_encoder.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.pkl")
    sample_output_path = os.path.join(output_dir, "sample_text_embeddings.parquet")
    full_output_path = os.path.join(output_dir, "full_text_embeddings.parquet")

    # --- Step 1: Sample data and build vocabulary ---
    print(f"\nStep 1: Sampling data and building vocabulary")
    try:
        subsample_size = min(subsample_size, len(df))
        df_sample = df.sample(n=subsample_size, random_state=random_state)
        print(f"Training embeddings on {len(df_sample)} samples")

        train_texts = df_sample[text_column].astype(str).tolist()
        if not train_texts:
            raise ValueError("No text data found in the subsample.")

        vocab = build_vocab(train_texts)

        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary with {len(vocab)} tokens saved to {vocab_path}")

    except KeyError:
        print(f"Error: Text column '{text_column}' not found in DataFrame.")
        return None
    except Exception as e:
        print(f"Error during sampling/vocab building: {e}")
        return None

    # --- Step 2: Create dataset and train model ---
    print("\nStep 2: Training model on subsample")
    try:
        pad_idx = vocab["<pad>"]
        dataset = TextDataset(train_texts, vocab)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        model = models.TextEncoder(len(vocab), embed_dim, hidden_dim, pad_idx=pad_idx).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() # Using MSELoss with dummy targets for unsupervised embedding learning

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                dummy_targets = torch.zeros_like(output, device=device)
                loss = criterion(output, dummy_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        # Save the trained model state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # --- Step 3: Get embeddings for the subsample (for validation/saving) ---
    print("\nStep 3: Generating embeddings for subsample (validation)")
    try:
        model.eval()
        sample_embeddings_list = []
        # Recreate dataloader for the sample data (no shuffle)
        sample_dataset = TextDataset(train_texts, vocab)
        sample_dataloader = DataLoader(sample_dataset, batch_size=infer_batch_size, shuffle=False, collate_fn=collate_fn)
        with torch.no_grad():
            for batch in sample_dataloader:
                batch = batch.to(device)
                embeddings = model(batch)
                sample_embeddings_list.append(embeddings.cpu().numpy())

        sample_embeddings_np = np.vstack(sample_embeddings_list)
        sample_embeddings_df = pd.DataFrame(
            sample_embeddings_np,
            index=df_sample.index, # Use index from df_sample
            columns=[f"embed_{i}" for i in range(sample_embeddings_np.shape[1])]
        )
        # Save Sample Embeddings
        sample_embeddings_df.to_parquet(sample_output_path)
        print(f"Sample embeddings saved to {sample_output_path}")

    except Exception as e:
        print(f"Error generating/saving sample embeddings: {e}")
        # Continue to generate full embeddings even if sample saving fails

    # --- Step 4: Apply trained model to full dataset for embeddings ---
    print("\nStep 4: Applying model to full dataset")
    try:
        # Ensure model is in eval mode
        model.eval()

        all_texts = df[text_column].astype(str).tolist()
        full_dataset = TextDataset(all_texts, vocab)
        full_dataloader = DataLoader(
            full_dataset, batch_size=infer_batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_embeddings_list = []
        with torch.no_grad():
            for batch in tqdm(full_dataloader, desc="Generating embeddings"):
                batch = batch.to(device)
                embedding_batch = model(batch)
                all_embeddings_list.append(embedding_batch.cpu().numpy())

        all_embeddings_np = np.vstack(all_embeddings_list)
        all_embeddings_df = pd.DataFrame(
            all_embeddings_np,
            index=df.index, # Use the original index from the input df
            columns=[f"embed_{i}" for i in range(all_embeddings_np.shape[1])]
        )

        print(f"\nEmbeddings generated for all {len(all_embeddings_df)} samples")
        print(f"Embeddings dimension: {all_embeddings_np.shape[1]}")

        # Save Full Embeddings
        all_embeddings_df.to_parquet(full_output_path)
        print(f"Embeddings saved to {sample_output_path} and {full_output_path}")

        print("--- LSTM Embedding Generation Complete ---")
        return all_embeddings_df

    except KeyError:
        print(f"Error: Text column '{text_column}' not found in DataFrame during inference.")
        return None
    except Exception as e:
        print(f"Error during embedding generation/saving: {e}")
        return None