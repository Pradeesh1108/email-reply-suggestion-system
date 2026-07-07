# Use official Python slim image
FROM python:3.12-slim

# Hugging Face Spaces requires running as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install uv for fast package management
RUN pip install --user --no-cache-dir uv

# Copy project specification files
COPY --chown=user:user pyproject.toml uv.lock ./

# Sync dependencies (creates .venv)
RUN uv sync --no-dev

# Copy application code
COPY --chown=user:user . .

# Ensure the dataset is built before starting
RUN uv run backend/scripts/build_dataset.py

# Expose port 7860 as required by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
