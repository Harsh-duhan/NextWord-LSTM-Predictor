# LSTM Next Word Prediction

A Streamlit application that generates the most likely next word from an LSTM language model trained with PyTorch.

The repository includes processed training data and a trained model artifact, so the prediction interface can run immediately.

## Requirements

- Python 3.11
- pip
- Docker Desktop (optional, for the containerized workflow)

## Run locally

Create and activate a virtual environment, install the pinned dependencies, then start Streamlit:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the address printed by Streamlit, normally `http://localhost:8501`.

If you need your own dataset for training model, you can use pdf_text_extractor preprocessing file which gives you a well crafted ready to use dataset in processed_data folder. Otherwise use the already available dataset with a huge corpus of 830K words.

## Train the model

The default command uses `processed_data/X.npy`, `processed_data/y.npy`, and `processed_data/word_index.json`. It writes the trained model to `artifacts/next_word_model.pt`.

```powershell
python train_model.py
```

The default training configuration uses a two-layer LSTM with 128-dimensional embeddings, 256 hidden units, 0.3 dropout, and gradient clipping. Override any setting when needed:

```powershell
python train_model.py --epochs 20 --embedding-dim 256 --hidden-dim 512 --num-layers 2 --dropout 0.3 --learning-rate 0.0005
```

When using `processed_data/X.npy`, its existing sequence width determines the context size. Recreate those arrays from the larger corpus to use a longer context window.

To train from a supported CSV, TXT, JSON, or pickle file instead:


```powershell
python train_model.py --data "your_preprocessed_file.csv" --epochs 10
```

## Run with Docker

### Pull from Docker Hub (recommended)

The image is published to Docker Hub as `harshduhan/nextword-lstm-predictor`:

```powershell
docker pull harshduhan/nextword-lstm-predictor:latest
docker run --rm -p 8501:8501 harshduhan/nextword-lstm-predictor:latest
```

Then open `http://localhost:8501`.

### Build locally

Build the image from the project folder and tag it to match the Hub name:

```powershell
docker build -t harshduhan/nextword-lstm-predictor:latest .
docker run --rm -p 8501:8501 harshduhan/nextword-lstm-predictor:latest
```

### Push to Docker Hub

```powershell
docker login
docker push harshduhan/nextword-lstm-predictor:latest
```

## Project layout

- `app.py`: Streamlit prediction interface.
- `model.py`: LSTM network, tokenization, and prediction helpers.
- `train_model.py`: model training script.
- `data_loader.py`: input readers for raw text datasets.
- `processed_data/`: prepared training arrays and vocabulary.
- `artifacts/next_word_model.pt`: trained model loaded by the app.
