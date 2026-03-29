# ============================================================
# Stage 1 — Base image
# Python 3.11 slim keeps the image small
# ============================================================
FROM python:3.11-slim

# ============================================================
# Stage 2 — System dependencies
# tesseract-ocr: OCR engine for PDF processing
# poppler-utils: pdf2image dependency (pdfinfo, pdftoppm)
# libgl1: required by some Pillow/image operations
# build-essential + git: required for some Python package builds
# ============================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 3 — Install Poetry
# ============================================================
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${POETRY_VENV}/bin:${PATH}"

# ============================================================
# Stage 4 — Install Python dependencies
# ============================================================
WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# ============================================================
# Stage 5 — Application code (changes most often — last)
# ============================================================
COPY . .

# ============================================================
# Runtime configuration
# ============================================================
EXPOSE 8000

ENV TESSERACT_PATH=/usr/bin/tesseract
ENV POPPLER_PATH=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PAGEINDEX_DIR=/app/pageocr
ENV UPLOAD_DIR=/app/pageocr/uploads
ENV CHROMA_DIR=/app/pageocr/chroma_db

# ============================================================
# Start command — runs from project root
# ============================================================
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]