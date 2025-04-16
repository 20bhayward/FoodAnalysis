# NewDesign/src/lstm_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm
import os

# Direct import from models
import models

# === Dataset Definition for Autoencoder ===
class RecipeTextDatasetAE(Dataset):
    """
    Dataset specific to the LSTM Autoencoder experiment.
    Includes vocabulary size limiting and tailored text cleaning.
    """
    def __init__(self, texts, vocab=None, max_len=100, max_vocab_size=15000):
        self.max_len = max_len
        # Only limit vocab size if it needs to be built
        if vocab is None:
            self.max_vocab_size = max_vocab_size
            self.texts = [self._clean_text(t) for t in texts]
            self.vocab = self._build_vocab(self.texts, self.max_vocab_size)
        else:
            # Still need cleaning even with provided vocab
            self.texts = [self._clean_text(t) for t in texts]
            self.vocab = vocab
            # Set based on provided vocab
            self.max_vocab_size = len(vocab)

        self.encoded = [self._encode(t) for t in self.texts]

    def _clean_text(self, text):
        """Applies cleaning rules defined for the Autoencoder experiment."""
        if not isinstance(text, str): return []
        measures = ['teaspoon', 't','tsp','tablespoon','T','tbsp','cup','c','pint','p','quart','q','gallon','g','oz','ounce','fl','fluid','lb','pound','mg','milligram','gram','kg','kilogram']
        words_to_remove = ['fresh', 'oil', 'a','red','bunch','green','white','black','yellow','large','small','medium','diced','chopped','sliced','minced','crushed','grated','peeled','seeded','cooked','uncooked','whole','halved','quartered','cubed','shredded','drained','rinsed','trimmed','divided','beaten','softened','melted','packed','dried','to','taste','for','serving','optional','as','needed','and','more','or','less','cut','into','strips','lengthwise','crosswise','thinly','thickly','sliced','canned','frozen','thawed','at','room','temperature','temp']
        remove_set = set(measures + words_to_remove).union(ENGLISH_STOP_WORDS)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        return [t for t in tokens if t not in remove_set and len(t) > 1]

    def _build_vocab(self, texts, max_vocab_size):
        """Builds vocab from token lists, limited to max_vocab_size."""
        word_freq = Counter(token for tokens in texts for token in tokens)

        limit = max(1, max_vocab_size - 2) if max_vocab_size is not None else None
        most_common_words = [word for word, count in word_freq.most_common(limit)]

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word in most_common_words:
            vocab[word] = len(vocab)

        print(f"Built vocabulary with {len(vocab)} tokens (limited from {len(word_freq)} unique words)")
        return vocab

    def _encode(self, tokens):
        """Encodes token list to tensor, padding/truncating included."""
        encoded = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        encoded = encoded[:self.max_len]
        padded = encoded + [self.vocab["<PAD>"]] * max(0, self.max_len - len(encoded))
        tensor = torch.tensor(padded, dtype=torch.long)
        # Target is the same as input for AE training
        return tensor, tensor

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]
# ===========================================

# === Training Function for Autoencoder ===
def train_autoencoder(
    model: models.LSTMAutoencoder,
    dataloader: DataLoader,
    vocab_size: int,
    checkpoint_path: str,
    device: str = 'cpu',
    epochs: int = 10,
    learning_rate: float = 1e-3
    ):
    """Trains the LSTM Autoencoder model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n--- Training LSTM Autoencoder ---")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Using device: {device}")
    print(f"Saving final checkpoint to: {checkpoint_path}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for inputs, targets in batch_iterator:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Reshape for CrossEntropyLoss: (Batch*SeqLen, VocabSize), (Batch*SeqLen)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_iterator.set_postfix(Loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved to {checkpoint_path}")

    print(f"Autoencoder model training finished. Best model saved to {checkpoint_path}")
    print(f"--- LSTM Autoencoder Training Complete ---")
    return model