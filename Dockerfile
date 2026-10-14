# syntax=docker/dockerfile:1
# Slim CPU-only image serving the BijliPredict AI FastAPI app.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install uv (fast, reproducible installs straight from the lockfile).
RUN pip install --no-cache-dir uv

# Dependency layer: install from the lock first so source edits don't bust it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project

# Application source. artifacts/ (best_model.pt, scaler.joblib, calibration.json)
# must be present locally — run `bijli train` first, or bind-mount them at run:
#   docker run -v "$PWD/artifacts:/app/artifacts:ro" -p 8000:8000 bijli-predict-ai
COPY . .
RUN uv sync --frozen --no-dev --extra serve

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
