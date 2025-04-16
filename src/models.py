# NewDesign/src/models.py

import torch
import torch.nn as nn
import tensorflow as tf # Keep TF import for Keras components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# --- LSTM Encoder ---
class TextEncoder(nn.Module):
    """LSTM Encoder to generate text embeddings."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """Forward pass: Embed -> LSTM -> Final Hidden State."""
        embedded = self.embedding(x)
        # LSTM output: sequence_output, (last_hidden_state, last_cell_state)
        _, (hidden, _) = self.lstm(embedded)
        # Return the hidden state of the last layer (batch_size, hidden_dim)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim) -> access last layer [ -1]
        return hidden[-1]

# --- LSTM Autoencoder ---
class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder model.
    The encoder compresses the input sequence into hidden/cell states.
    The decoder takes the final encoder state and attempts to reconstruct the sequence.
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, pad_idx=0):
        """
        Initializes the Autoencoder layers.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dim (int): The dimension of the LSTM hidden states.
            pad_idx (int): The index of the padding token in the vocabulary.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        """
        Full autoencoder forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        embedded = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)

        # Use final encoder hidden state as context, repeat for each decoder step
        # hidden[-1] gives shape (batch_size, hidden_dim)
        decoder_input_context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # Initial state for decoder is the final state of the encoder
        decoder_initial_state = (hidden, cell)

        # Decoder processes the context sequence
        decoded_output, _ = self.decoder(decoder_input_context, decoder_initial_state)
        # Project decoder hidden states to vocabulary size
        output_logits = self.output(decoded_output)

        return output_logits

    def encode(self, x):
        """
        Encoder-only forward pass to get the final hidden state representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            torch.Tensor: Final hidden state tensor of shape (batch_size, hidden_dim).
        """
        embedded = self.embedding(x)
        # LSTM handles padding implicitly if pad_idx is set in embedding.
        _, (hidden, _) = self.encoder(embedded)
        # Return the last layer's hidden state, removing the layer dimension.
        return hidden[-1]

    def decode_logits(self, hidden_state, seq_len):
         """
         Decodes from a hidden state to generate output logits for a sequence.
         (Helper for qualitative check/generation tasks).

         Args:
             hidden_state (torch.Tensor): Shape (batch_size, hidden_dim) or (1, batch_size, hidden_dim)
             seq_len (int): The desired output sequence length.

         Returns:
             torch.Tensor: Output logits tensor of shape (batch_size, seq_len, vocab_size).
         """
         batch_size = hidden_state.size(0) if hidden_state.dim() == 2 else hidden_state.size(1)
         # Ensure hidden state has layer dimension if needed by LSTM (shape: num_layers, batch, hidden_dim)
         if hidden_state.dim() == 2:
              hidden = hidden_state.unsqueeze(0) # Add layer dim: (1, batch_size, hidden_dim)
         else:
              hidden = hidden_state

         # Create dummy cell state matching hidden state dimensions
         cell = torch.zeros_like(hidden) # (1, batch_size, hidden_dim)
         decoder_initial_state = (hidden, cell)

         # Use hidden state as context for each step, shape (batch_size, seq_len, hidden_dim)
         decoder_input_context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

         # Decode
         decoded_output, _ = self.decoder(decoder_input_context, decoder_initial_state)
         output_logits = self.output(decoded_output)
         return output_logits


# --- Keras Neural Network ---
def build_neural_network(input_dim, output_dim):
    """Builds the Keras Sequential model for multi-label classification."""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid') # Sigmoid for multi-label binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Appropriate loss for multi-label sigmoid output
        metrics=['accuracy'] # Keras 'accuracy' for multilabel often means subset accuracy (exact match)
    )
    return model