# Entropy Analysis - Docker Image
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Builder with uv
# ==============================================================================
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --no-dev

# ==============================================================================
# Stage 2: Runtime (optimized)
# ==============================================================================
FROM python:3.12-slim AS runtime

# Install runtime dependencies and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports
# 8501 - Streamlit dashboard
# 8000 - FastAPI server
EXPOSE 8501 8000

# Default command: run Streamlit dashboard
CMD ["streamlit", "run", "/app/src/entropy_analysis/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

