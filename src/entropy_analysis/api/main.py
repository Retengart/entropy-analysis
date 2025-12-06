"""
FastAPI application for entropy analysis.

Provides REST API endpoints for text analysis, comparison, and visualization.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.normalize import Alphabet
from entropy_analysis.models.schemas import (
    AnalyzerConfigRequest,
    BatchTextsRequest,
    CompareTextsRequest,
    ComparisonResponse,
    RollingEntropyRequest,
    RollingEntropyResponse,
    TextAnalysisResponse,
    UploadedTextRequest,
)

# Global analyzer instance
_analyzer: TextAnalyzer | None = None


def get_analyzer() -> TextAnalyzer:
    """Get or create the global analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TextAnalyzer.create()
    return _analyzer


def set_analyzer(analyzer: TextAnalyzer) -> None:
    """Set the global analyzer."""
    global _analyzer
    _analyzer = analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    # Initialize default analyzer
    get_analyzer()
    yield
    # Cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Entropy Analysis API",
        description="Comprehensive entropy analysis toolkit for text analysis",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


app = create_app()


def _result_to_response(result: Any) -> TextAnalysisResponse:
    """Convert TextAnalysisResult to API response."""
    from entropy_analysis.models.schemas import (
        BootstrapResponse,
        EnhancedMetricsResponse,
        LetterDistribution,
        NormalizedEntropyResponse,
    )

    letter_dist = [
        LetterDistribution(
            rank=s.rank,
            letter=s.letter,
            count=s.count,
            probability=s.probability,
            p_log2=s.p_log2,
        )
        for s in result.letter_stats
    ]

    normalized = None
    if result.normalized_entropy:
        normalized = NormalizedEntropyResponse(
            h_normalized=result.normalized_entropy.h_normalized,
            entropy_per_word=result.normalized_entropy.entropy_per_word,
            efficiency=result.normalized_entropy.efficiency,
        )

    bootstrap = None
    if result.bootstrap:
        bootstrap = BootstrapResponse(
            estimate=result.bootstrap.estimate,
            std_error=result.bootstrap.std_error,
            ci_lower=result.bootstrap.ci_lower,
            ci_upper=result.bootstrap.ci_upper,
            confidence_level=result.bootstrap.confidence_level,
            n_bootstrap=result.bootstrap.n_bootstrap,
        )

    enhanced = None
    if result.enhanced:
        enhanced = EnhancedMetricsResponse(
            perplexity=result.enhanced.perplexity,
            alphabet_utilization=result.enhanced.alphabet_utilization,
            evenness=result.enhanced.evenness,
            herfindahl_index=result.enhanced.herfindahl_index,
            uniformity_distance=result.enhanced.uniformity_distance,
            uniqueness_ratio=result.enhanced.uniqueness_ratio,
            redundancy=result.enhanced.redundancy,
            renyi_0=result.enhanced.renyi_0,
            renyi_2=result.enhanced.renyi_2,
            renyi_inf=result.enhanced.renyi_inf,
        )

    return TextAnalysisResponse(
        n_words=result.n_words,
        n_unique_letters=result.n_unique_letters,
        shannon_entropy=result.shannon_entropy,
        mean_rank=result.mean_rank,
        std_rank=result.std_rank,
        letter_distribution=letter_dist,
        normalized_entropy=normalized,
        miller_madow_entropy=result.miller_madow_entropy,
        bootstrap=bootstrap,
        simpson_index=result.simpson_index,
        gini_simpson_index=result.gini_simpson_index,
        zipf_alpha=result.zipf_alpha,
        zipf_r_squared=result.zipf_r_squared,
        compression_ratio=result.compression_ratio,
        enhanced=enhanced,
        yules_k=result.yules_k,
        mtld=result.mtld,
        mattr=result.mattr,
        gries_dp=result.gries_dp,
        bigram_entropy=result.bigram_entropy,
        trigram_entropy=result.trigram_entropy,
        burstiness=result.burstiness,
        hurst_exponent=result.hurst_exponent,
        zipf_mandelbrot_alpha=result.zipf_mandelbrot_alpha,
        zipf_mandelbrot_beta=result.zipf_mandelbrot_beta,
        pierrehumbert_beta=result.pierrehumbert_beta,
        source_name=result.source_name,
        alphabet_size=result.alphabet_size,
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Entropy Analysis API",
        "version": "2.0.0",
        "description": "Comprehensive entropy analysis toolkit",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/configure")
async def configure_analyzer(config: AnalyzerConfigRequest):
    """Configure the analyzer with custom settings."""
    try:
        alphabet = Alphabet(config.alphabet)
        analyzer = TextAnalyzer.create(
            alphabet=alphabet,
            custom_letters=config.custom_letters,
            keep_yo=config.keep_yo,
            keep_j=config.keep_j,
            min_token_len=config.min_token_len,
        )
        set_analyzer(analyzer)
        return {"status": "configured", "alphabet": config.alphabet}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: UploadedTextRequest):
    """
    Analyze a single text.

    Returns comprehensive entropy analysis including:
    - Shannon entropy
    - Letter distribution
    - Normalized metrics
    - Diversity indices
    - Zipf analysis
    """
    analyzer = get_analyzer()

    result = analyzer.analyze(
        text=request.text,
        source_name=request.name,
        include_bootstrap=request.include_bootstrap,
        bootstrap_iterations=request.bootstrap_iterations,
        include_complexity=request.include_complexity,
        include_advanced_metrics=request.include_advanced_metrics,
    )

    return _result_to_response(result)


@app.post("/api/v1/analyze/file", response_model=TextAnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),  # noqa: B008
    include_bootstrap: bool = False,
    include_complexity: bool = False,
    include_advanced_metrics: bool = False,
):
    """Analyze an uploaded text file."""
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        # Try other encodings
        try:
            text = content.decode("cp1251")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Could not decode file") from e

    analyzer = get_analyzer()
    result = analyzer.analyze(
        text=text,
        source_name=file.filename,
        include_bootstrap=include_bootstrap,
        include_complexity=include_complexity,
        include_advanced_metrics=include_advanced_metrics,
    )

    return _result_to_response(result)


@app.post("/api/v1/compare", response_model=ComparisonResponse)
async def compare_texts(request: CompareTextsRequest):
    """
    Compare two texts using divergence metrics.

    Returns:
    - KL divergence (both directions)
    - Jensen-Shannon divergence
    - Cosine similarity
    """
    analyzer = get_analyzer()

    comparison = analyzer.compare(
        text1=request.text1,
        text2=request.text2,
        name1=request.name1,
        name2=request.name2,
    )

    return ComparisonResponse(
        text1_name=comparison.text1_name,
        text2_name=comparison.text2_name,
        kl_divergence_p_q=comparison.kl_divergence_p_q,
        kl_divergence_q_p=comparison.kl_divergence_q_p,
        js_divergence=comparison.js_divergence,
        cosine_similarity=comparison.cosine_similarity,
    )


@app.post("/api/v1/rolling", response_model=RollingEntropyResponse)
async def rolling_entropy(request: RollingEntropyRequest):
    """Calculate rolling entropy through the text."""
    analyzer = get_analyzer()

    result = analyzer.rolling_entropy(
        text=request.text,
        window_size=request.window_size,
        step_size=request.step_size,
    )

    return RollingEntropyResponse(
        positions=result.positions,
        entropies=result.entropies,
        window_size=result.window_size,
        mean_entropy=result.mean_entropy,
        std_entropy=result.std_entropy,
        min_entropy=result.min_entropy,
        max_entropy=result.max_entropy,
    )


@app.post("/api/v1/batch")
async def analyze_batch(request: BatchTextsRequest):
    """Analyze multiple texts at once."""
    from entropy_analysis.models.schemas import (
        BatchAnalysisResponse,
        CorrelationResult,
        ExtendedStatsResponse,
    )

    analyzer = get_analyzer()

    texts = [(t.name or f"text_{i}", t.text) for i, t in enumerate(request.texts)]
    batch_result = analyzer.analyze_batch(texts, include_bootstrap=request.include_bootstrap)

    results = [_result_to_response(r) for _, r in batch_result.results]

    extended = None
    if batch_result.extended_stats:
        es = batch_result.extended_stats
        extended = ExtendedStatsResponse(
            mean=es.mean,
            median=es.median,
            std_dev=es.std_dev,
            variance=es.variance,
            q1=es.q1,
            q3=es.q3,
            iqr=es.iqr,
            coefficient_of_variation=es.coefficient_of_variation,
            skewness=es.skewness,
            kurtosis=es.kurtosis,
            min_value=es.min_value,
            max_value=es.max_value,
        )

    correlation = None
    if batch_result.correlation is not None:
        correlation = CorrelationResult(
            n_samples=len(results),
            correlation=batch_result.correlation,
            slope=batch_result.correlation_slope or 0,
            intercept=batch_result.correlation_intercept or 0,
            r_squared=batch_result.correlation_r_squared or 0,
            p_value=batch_result.correlation_p_value or 1,
        )

    return BatchAnalysisResponse(
        results=results,
        extended_stats=extended,
        correlation=correlation,
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
