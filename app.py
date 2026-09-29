"""Streamlit interface for the student LSTM next-word prediction project."""

from pathlib import Path

import streamlit as st
import torch

from model import NextWordLSTM, sample_next_word, tokenize


PROJECT_FOLDER = Path(__file__).parent
MODEL_PATH = PROJECT_FOLDER / "artifacts" / "next_word_model.pt"


@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = NextWordLSTM(
        vocab_size=len(checkpoint["index_to_word"]),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, device


st.set_page_config(page_title="Next Word Predictor", page_icon="N", layout="centered")
st.title("Next Word Predictor")

if not MODEL_PATH.exists():
    st.info("Train the model first: python train_model.py")
    st.stop()

model, checkpoint, device = load_trained_model()
with st.sidebar:
    st.subheader("Model")
    st.write(f"Vocabulary: {len(checkpoint['index_to_word'])} words")
    st.write(f"Context: {checkpoint['context_size']} words")
    output_length = st.number_input(
        "Number of output words",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.4,
        max_value=1.2,
        value=0.8,
        step=0.1,
    )
    top_k = st.slider("Top-k choices", min_value=2, max_value=20, value=8)
    repetition_penalty = st.slider(
        "Repetition penalty",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
    )

sentence = st.text_area("Enter a sentence", value="machine learning is", height=150)

if st.button("Predict next word", type="primary"):
    if not tokenize(sentence):
        st.warning("Please enter at least one word.")
    else:
        current_text = sentence
        generated_words = []

        for _ in range(output_length):
            next_word = sample_next_word(
                model,
                current_text,
                checkpoint["word_to_index"],
                checkpoint["index_to_word"],
                device,
                context_size=checkpoint["context_size"],
                top_k=top_k,
                temperature=temperature,
                recent_words=generated_words,
                repetition_penalty=repetition_penalty,
            )
            if not next_word:
                break

            generated_words.append(next_word)
            current_text += f" {next_word}"

        st.subheader("Prediction")
        st.write(current_text)
