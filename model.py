"""Small LSTM model and helper functions for next-word prediction."""

from collections import Counter
import re

import torch
from torch import nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    """Turn a sentence into simple lowercase word tokens."""
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def make_vocabulary(sentences, max_words=5000):
    """Create word-to-number and number-to-word dictionaries."""
    all_words = []
    for sentence in sentences:
        all_words.extend(tokenize(sentence))

    common_words = Counter(all_words).most_common(max_words - 2)
    index_to_word = [PAD_TOKEN, UNK_TOKEN] + [word for word, _ in common_words]
    word_to_index = {word: index for index, word in enumerate(index_to_word)}
    return word_to_index, index_to_word


class NextWordLSTM(nn.Module):
    """Embedding -> stacked LSTM -> Linear next-word model."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=3,
        dropout=0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, words, lengths):
        embedded_words = self.embedding(words)
        lstm_output, _ = self.lstm(embedded_words)
        last_positions = lengths - 1
        row_numbers = torch.arange(words.size(0), device=words.device)
        final_output = lstm_output[row_numbers, last_positions]
        return self.output(final_output)


def sample_next_word(
    model,
    text,
    word_to_index,
    index_to_word,
    device,
    context_size=5,
    top_k=8,
    temperature=0.8,
    recent_words=None,
    repetition_penalty=1.2,
):
    """Sample one likely next word while discouraging recent repetition."""
    words = tokenize(text)
    if not words:
        return None

    word_numbers = [word_to_index.get(word, 1) for word in words[-context_size:]]
    input_tensor = torch.tensor([word_numbers], dtype=torch.long, device=device)
    lengths = torch.tensor([len(word_numbers)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        scores = model(input_tensor, lengths)[0].clone()
        scores[0] = float("-inf")
        scores[1] = float("-inf")

        for word in set((recent_words or [])[-20:]):
            index = word_to_index.get(word)
            if index is not None:
                scores[index] -= repetition_penalty

        scores = scores / temperature
        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(index_to_word) - 2)
        )
        probabilities = torch.softmax(top_scores, dim=0)
        selected = torch.multinomial(probabilities, num_samples=1).item()

    return index_to_word[top_indices[selected].item()]
