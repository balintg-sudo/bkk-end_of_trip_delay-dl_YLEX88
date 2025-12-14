FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN export TORCH=$(python -c "import torch; print(torch.__version__)") && \
    export TORCH=${TORCH%+*} && \
    pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p log data models

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD python src/00-download.py && \
    python src/01-data-preprocessing.py && \
    python src/02-training.py && \
    python src/03-evaluation.py
