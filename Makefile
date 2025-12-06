# Makefile for Entropy Analysis project

.PHONY: help install dev test lint format clean docker-build docker-run dashboard api cli

# Default target
help:
	@echo "🎓 Entropy Analysis - Makefile commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install dependencies with uv"
	@echo "  make dev           - Install dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make example       - Run example script"
	@echo "  make demo-viz      - Demo all 9 visualization types"
	@echo ""
	@echo "Run:"
	@echo "  make dashboard     - Start Streamlit dashboard"
	@echo "  make api           - Start FastAPI server"
	@echo "  make cli           - Show CLI help"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run dashboard in Docker"
	@echo "  make docker-up     - Start all services (docker-compose)"
	@echo "  make docker-down   - Stop all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean cache files"

# Install dependencies
install:
	uv sync

# Install dev dependencies
dev:
	uv sync --all-extras

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=src/entropy_analysis --cov-report=html --cov-report=term

# Lint code
lint:
	uvx ruff check src/ tests/ || true

# Format code
format:
	uvx ruff check --fix src/ tests/ || true

# Run example
example:
	uv run python example_usage.py

# Demo all visualizations
demo-viz:
	uv run python demo_visualizations.py
	@echo ""
	@echo "✅ All charts created in demo_output/"
	@echo "💡 Open them in browser to see interactive plots!"

# Start dashboard
dashboard:
	@echo "🎨 Starting Streamlit dashboard..."
	@echo "📊 Open in browser: http://localhost:8501"
	uv run streamlit run src/entropy_analysis/dashboard/app.py

# Start API server
api:
	@echo "🚀 Starting FastAPI server..."
	@echo "📚 Documentation: http://localhost:8000/docs"
	uv run uvicorn entropy_analysis.api.main:app --reload

# Show CLI help
cli:
	uv run entropy-analysis --help

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t entropy-analysis:latest .

# Run dashboard in Docker
docker-run:
	@echo "🎨 Running dashboard in Docker..."
	@echo "📊 Open in browser: http://localhost:8501"
	docker run --rm -p 8501:8501 entropy-analysis:latest

# Start all services with docker-compose
docker-up:
	@echo "🐳 Starting all services..."
	docker-compose up -d
	@echo ""
	@echo "✅ Services running:"
	@echo "   📊 Dashboard: http://localhost:8501"
	@echo "   🚀 API: http://localhost:8000/docs"

# Stop all services
docker-down:
	@echo "🛑 Stopping all services..."
	docker-compose down

# Production deployment
docker-prod:
	@echo "🚀 Starting production deployment..."
	docker-compose -f docker-compose.production.yml up -d

# Clean cache files
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage
	@echo "✅ Clean complete"

# Analyze sample text from old project
analyze-pushkin:
	uv run entropy-analysis analyze ../entropy-analysis/text/pushkin.txt --verbose

# Batch analyze old project texts
analyze-batch:
	uv run entropy-analysis batch ../entropy-analysis/text/ --csv batch_results.csv

# Compare Pushkin vs Lermontov
compare-authors:
	@if [ -f ../entropy-analysis/text/pushkin.txt ] && [ -f ../entropy-analysis/text/lermontov_new.txt ]; then \
		uv run entropy-analysis compare \
			../entropy-analysis/text/pushkin.txt \
			../entropy-analysis/text/lermontov_new.txt; \
	else \
		echo "❌ Files not found. Please check paths."; \
	fi

# Full workflow demo
demo:
	@echo "🎬 Running full demo..."
	@make example
	@echo ""
	@echo "📄 Analyzing Pushkin..."
	@make analyze-pushkin || true
	@echo ""
	@echo "✅ Demo complete!"

# Check if everything works
check: test lint
	@echo ""
	@echo "✅ All checks passed!"

