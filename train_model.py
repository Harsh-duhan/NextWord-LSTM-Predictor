import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from data_loader import find_dataset, load_sentences
from model import NextWordLSTM, make_vocabulary, tokenize


class NextWordDataset(Dataset):
    def __init__(self, sentences, word_to_index, context_size, max_examples):
        self.examples = []
        for sentence in sentences:
            words = tokenize(sentence)
            for target_position in range(1, len(words)):
                start = max(0, target_position - context_size)
                context = [word_to_index.get(word, 1) for word in words[start:target_position]]
                target = word_to_index.get(words[target_position], 1)
                self.examples.append((torch.tensor(context), target))
                if len(self.examples) >= max_examples:
                    return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PreprocessedDataset(Dataset):
    """Use the X.npy inputs and y.npy labels created during preprocessing."""

    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def load_preprocessed_data(data_folder, max_examples):
    """Load the existing NumPy arrays and vocabulary from processed_data."""
    inputs = np.load(data_folder / "X.npy")[:max_examples]
    targets = np.load(data_folder / "y.npy")[:max_examples]
    with open(data_folder / "word_index.json", encoding="utf-8") as file:
        word_to_index = {word: int(index) for word, index in json.load(file).items()}

    vocab_size = max(word_to_index.values()) + 1
    index_to_word = ["<pad>"] * vocab_size
    for word, index in word_to_index.items():
        index_to_word[index] = word
    return inputs, targets, word_to_index, index_to_word


def make_sentence_batch(batch):
    """Pad different-length sentence contexts when raw text is used."""
    contexts, targets = zip(*batch)
    lengths = torch.tensor([len(context) for context in contexts], dtype=torch.long)
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    return padded_contexts, lengths, torch.tensor(targets, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to the preprocessed data file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=5000)
    parser.add_argument("--max-examples", type=int, default=200000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    arguments = parser.parse_args()

    if arguments.num_layers < 1:
        parser.error("--num-layers must be at least 1")
    if not 0 <= arguments.dropout < 1:
        parser.error("--dropout must be between 0 (inclusive) and 1 (exclusive)")
    if arguments.gradient_clip <= 0:
        parser.error("--gradient-clip must be greater than 0")

    project_folder = Path(__file__).parent
    processed_folder = project_folder / "processed_data"

    if (processed_folder / "X.npy").exists() and (processed_folder / "y.npy").exists():
        inputs, targets, word_to_index, index_to_word = load_preprocessed_data(
            processed_folder, arguments.max_examples
        )
        dataset = PreprocessedDataset(inputs, targets)
        sequence_length = inputs.shape[1]
        source_name = "processed_data/X.npy and y.npy"
        uses_fixed_length_inputs = True
    else:
        data_path = Path(arguments.data) if arguments.data else find_dataset(project_folder)
        sentences = load_sentences(data_path)
        if not sentences:
            raise ValueError("The data file did not contain readable text sentences.")
        word_to_index, index_to_word = make_vocabulary(sentences, arguments.max_words)
        dataset = NextWordDataset(
            sentences, word_to_index, arguments.context_size, arguments.max_examples
        )
        sequence_length = arguments.context_size
        source_name = data_path.name
        uses_fixed_length_inputs = False

    if len(dataset) == 0:
        raise ValueError("At least one training example is needed.")

    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        collate_fn=None if uses_fixed_length_inputs else make_sentence_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordLSTM(
        len(index_to_word),
        embedding_dim=arguments.embedding_dim,
        hidden_dim=arguments.hidden_dim,
        num_layers=arguments.num_layers,
        dropout=arguments.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    print(f"Using {source_name}: {len(dataset)} examples")
    for epoch in range(arguments.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            if uses_fixed_length_inputs:
                contexts, targets = batch
                lengths = torch.full((len(contexts),), contexts.shape[1], dtype=torch.long)
            else:
                contexts, lengths, targets = batch
            contexts, lengths, targets = contexts.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(contexts, lengths), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{arguments.epochs} - loss: {total_loss / len(loader):.4f}")

    artifact_folder = project_folder / "artifacts"
    artifact_folder.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "context_size": sequence_length,
        "embedding_dim": arguments.embedding_dim,
        "hidden_dim": arguments.hidden_dim,
        "num_layers": arguments.num_layers,
        "dropout": arguments.dropout,
        "data_file": source_name,
    }, artifact_folder / "next_word_model.pt")
    print("Saved model to artifacts/next_word_model.pt")


if __name__ == "__main__":
    main()
