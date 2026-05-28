# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim keeps the image small (~130 MB) while matching a common
# production Python version.
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="Iris flower species classifier — RandomForestClassifier with scikit-learn"
LABEL version="1.0"

# ── System hygiene ────────────────────────────────────────────────────────────
# Disable .pyc files and enable unbuffered stdout/stderr so print() output
# appears immediately in docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer separately from
# your source code — rebuild is fast when only .py files change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY train.py   .
COPY predict.py .

# ── Create data directory ─────────────────────────────────────────────────────
# train.py writes CSV splits here; declare it so the path always exists.
RUN mkdir -p data

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Run train.py first (builds & saves the model), then predict.py (loads &
# runs inference). Both scripts exit with a non-zero code on error, so the
# container will fail loudly if anything goes wrong.
CMD ["sh", "-c", "python train.py && python predict.py"]
