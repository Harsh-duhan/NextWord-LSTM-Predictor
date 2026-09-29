"""Read a preprocessed text dataset from common beginner-friendly formats."""

import json
import pickle
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ["text", "sentence", "clean_text", "processed_text", "content"]


def find_dataset(folder):
    """Find the first supported dataset file in the project folder."""
    extensions = {".csv", ".txt", ".json", ".pkl", ".pickle"}
    for path in folder.iterdir():
        if path.suffix.lower() in extensions:
            return path
    raise FileNotFoundError("No supported data file was found in the project folder.")


def _strings_from_data(data):
    """Collect strings from a list, dictionary, pandas table, or single value."""
    if isinstance(data, str):
        return [data]
    if isinstance(data, pd.DataFrame):
        lower_names = {str(column).lower(): column for column in data.columns}
        for name in TEXT_COLUMNS:
            if name in lower_names:
                return data[lower_names[name]].dropna().astype(str).tolist()
        text_columns = data.select_dtypes(include=["object", "string"]).columns
        return data[text_columns[0]].dropna().astype(str).tolist() if len(text_columns) else []
    if isinstance(data, pd.Series):
        return data.dropna().astype(str).tolist()
    if isinstance(data, dict):
        for name in TEXT_COLUMNS:
            if name in data:
                return _strings_from_data(data[name])
        return [text for value in data.values() for text in _strings_from_data(value)]
    if isinstance(data, (list, tuple)):
        return [text for item in data for text in _strings_from_data(item)]
    return []


def load_sentences(path):
    """Load and clean non-empty text rows from a supported file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    elif suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as file:
            data = pickle.load(file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return [sentence.strip() for sentence in _strings_from_data(data) if sentence.strip()]
