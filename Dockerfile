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

# Copy dependency files first (for better layer caching)
# README.md is needed because it's referenced in pyproject.toml
COPY pyproject.toml uv.lock README.md ./

# Create virtual environment and install dependencies
# Use --frozen to use lock file and avoid re-resolving dependencies
# Mount cache for uv to reuse downloaded packages across builds
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy source code (after dependencies are installed for better caching)
COPY src/ ./src/

# Install the package in editable mode so it can be imported
RUN uv pip install -e . --no-deps

# ==============================================================================
# Stage 2: Runtime (optimized)
# ==============================================================================
FROM python:3.12-slim AS runtime

# Install runtime dependencies and curl for health checks
# Cache apt packages across builds using BuildKit cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code and project files needed for package installation
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/uv.lock /app/uv.lock

# Install uv in runtime stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install the package in editable mode using uv pip install
# Set VIRTUAL_ENV so uv knows which environment to use
ENV VIRTUAL_ENV=/app/.venv
RUN uv pip install -e . --no-deps

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
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

