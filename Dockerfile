
# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements.txt .

# Install CPU-only PyTorch + remaining deps in one layer
RUN pip install --user --no-warn-script-location \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="NextWord LSTM Predictor" \
      org.opencontainers.image.description="Streamlit app for next-word prediction using a PyTorch LSTM model" \
      org.opencontainers.image.authors="harshduhan" \
      org.opencontainers.image.source="https://github.com/Harsh-duhan/NextWord-LSTM-Predictor" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    # Streamlit configuration via env (avoids needing .streamlit/config.toml)
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy only the runtime files (see .dockerignore for exclusions)
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
