"""
Command-line interface for entropy analysis.

Usage:
    entropy-analysis analyze <file>
    entropy-analysis batch <directory>
    entropy-analysis compare <file1> <file2>
    entropy-analysis serve [--port PORT]
    entropy-analysis dashboard
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from entropy_analysis.core.analyze import TextAnalyzer
from entropy_analysis.core.normalize import Alphabet

app = typer.Typer(
    name="entropy-analysis",
    help="Comprehensive entropy analysis toolkit for text analysis",
    add_completion=False,
)
console = Console()


def create_analyzer(
    alphabet: str = "rus29",
    keep_yo: bool = False,
    keep_j: bool = False,
    min_token_len: int = 1,
) -> TextAnalyzer:
    """Create analyzer with specified options."""
    return TextAnalyzer.create(
        alphabet=Alphabet(alphabet),
        keep_yo=keep_yo,
        keep_j=keep_j,
        min_token_len=min_token_len,
    )


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to text file"),  # noqa: B008
    alphabet: str = typer.Option(
        "rus29", "--alphabet", "-a", help="Alphabet: rus29, rus33, custom (rus28 is supported for backward compatibility)"
    ),
    keep_yo: bool = typer.Option(False, "--keep-yo", help="Keep ё (don't replace with е)"),
    keep_j: bool = typer.Option(False, "--keep-j", help="Keep й (don't replace with и)"),
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Output JSON file"),  # noqa: B008
    output_csv: Path | None = typer.Option(None, "--csv", "-c", help="Output CSV file"),  # noqa: B008
    bootstrap: bool = typer.Option(False, "--bootstrap", "-b", help="Calculate bootstrap CI"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Analyze a single text file."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    # Read file
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file.read_text(encoding="cp1251")

    # Create analyzer
    analyzer = create_analyzer(alphabet, keep_yo, keep_j)

    # Analyze
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing...", total=None)
        result = analyzer.analyze(
            text,
            source_name=str(file),
            include_bootstrap=bootstrap,
        )

    # Display results
    console.print()
    console.print(f"[bold blue]📊 Analysis Results for:[/bold blue] {file.name}")
    console.print()

    # Metrics table
    metrics_table = Table(title="Key Metrics", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    metrics_table.add_row("N (words)", f"{result.n_words:,}")
    metrics_table.add_row("Unique letters", str(result.n_unique_letters))

    if result.shannon_entropy:
        metrics_table.add_row("Shannon entropy H", f"{result.shannon_entropy:.4f} bits")
    if result.mean_rank:
        metrics_table.add_row("Mean rank x̄", f"{result.mean_rank:.4f}")
    if result.std_rank:
        metrics_table.add_row("Std deviation σ", f"{result.std_rank:.4f}")
    if result.miller_madow_entropy:
        metrics_table.add_row("H (Miller-Madow)", f"{result.miller_madow_entropy:.4f}")
    if result.simpson_index is not None:
        metrics_table.add_row("Simpson Index", f"{result.simpson_index:.4f}")
    if result.zipf_alpha:
        metrics_table.add_row("Zipf α", f"{result.zipf_alpha:.3f}")

    console.print(metrics_table)

    # Bootstrap results
    if result.bootstrap:
        console.print()
        console.print(
            f"[bold]Bootstrap CI ({result.bootstrap.confidence_level * 100:.0f}%):[/bold] "
            f"[{result.bootstrap.ci_lower:.4f}, {result.bootstrap.ci_upper:.4f}]"
        )

    # Verbose: letter distribution
    if verbose:
        console.print()
        dist_table = Table(title="Letter Distribution", show_header=True)
        dist_table.add_column("Rank", justify="right")
        dist_table.add_column("Letter", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Probability", justify="right")

        for stat in result.letter_stats:
            if stat.count > 0:
                dist_table.add_row(
                    str(stat.rank), stat.letter, str(stat.count), f"{stat.probability:.4f}"
                )

        console.print(dist_table)

    # Output files
    if output_json:
        data = {
            "source": str(file),
            "n_words": result.n_words,
            "shannon_entropy": result.shannon_entropy,
            "mean_rank": result.mean_rank,
            "std_rank": result.std_rank,
            "letters": [
                {"rank": s.rank, "letter": s.letter, "count": s.count, "p": s.probability}
                for s in result.letter_stats
            ],
        }
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        console.print(f"\n[green]JSON saved to:[/green] {output_json}")

    if output_csv:
        df = analyzer.to_dataframe(result)
        df.write_csv(output_csv)
        console.print(f"[green]CSV saved to:[/green] {output_csv}")


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory with text files"),  # noqa: B008
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File glob pattern"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Search recursively"),
    alphabet: str = typer.Option("rus29", "--alphabet", "-a", help="Alphabet type"),
    output_csv: Path | None = typer.Option(None, "--csv", "-c", help="Output CSV file"),  # noqa: B008
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Output JSON file"),  # noqa: B008
):
    """Analyze all text files in a directory."""
    if not directory.exists():
        console.print(f"[red]Error:[/red] Directory not found: {directory}")
        raise typer.Exit(1)

    analyzer = create_analyzer(alphabet)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing files...", total=None)
        batch_result = analyzer.analyze_directory(
            directory,
            pattern=pattern,
            recursive=recursive,
        )

    console.print()
    console.print(f"[bold blue]📁 Batch Analysis:[/bold blue] {len(batch_result.results)} files")
    console.print()

    # Summary
    if batch_result.extended_stats:
        stats = batch_result.extended_stats
        summary_table = Table(title="Summary Statistics", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Mean H", f"{stats.mean:.4f}")
        summary_table.add_row("Median H", f"{stats.median:.4f}")
        summary_table.add_row("Std Dev", f"{stats.std_dev:.4f}")
        summary_table.add_row("Min H", f"{stats.min_value:.4f}")
        summary_table.add_row("Max H", f"{stats.max_value:.4f}")

        if batch_result.correlation is not None:
            summary_table.add_row("Correlation (N, H)", f"{batch_result.correlation:.4f}")
            summary_table.add_row("R²", f"{batch_result.correlation_r_squared:.4f}")

        console.print(summary_table)

    # Files table
    files_table = Table(title="Results by File", show_header=True)
    files_table.add_column("File", style="cyan", max_width=40)
    files_table.add_column("N", justify="right")
    files_table.add_column("H", justify="right")
    files_table.add_column("x̄", justify="right")
    files_table.add_column("σ", justify="right")

    for name, result in batch_result.results:
        files_table.add_row(
            Path(name).name,
            str(result.n_words),
            f"{result.shannon_entropy:.4f}" if result.shannon_entropy else "—",
            f"{result.mean_rank:.2f}" if result.mean_rank else "—",
            f"{result.std_rank:.2f}" if result.std_rank else "—",
        )

    console.print()
    console.print(files_table)

    # Output
    if output_csv:
        df = analyzer.batch_to_dataframe(batch_result)
        df.write_csv(output_csv)
        console.print(f"\n[green]CSV saved to:[/green] {output_csv}")

    if output_json:
        data = {
            "n_files": len(batch_result.results),
            "correlation": batch_result.correlation,
            "r_squared": batch_result.correlation_r_squared,
            "results": [
                {
                    "file": name,
                    "n_words": r.n_words,
                    "entropy": r.shannon_entropy,
                    "mean_rank": r.mean_rank,
                    "std_rank": r.std_rank,
                }
                for name, r in batch_result.results
            ],
        }
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        console.print(f"[green]JSON saved to:[/green] {output_json}")


@app.command()
def compare(
    file1: Path = typer.Argument(..., help="First text file"),  # noqa: B008
    file2: Path = typer.Argument(..., help="Second text file"),  # noqa: B008
    alphabet: str = typer.Option("rus29", "--alphabet", "-a", help="Alphabet type"),
):
    """Compare two text files."""
    for f in [file1, file2]:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    # Read files
    try:
        text1 = file1.read_text(encoding="utf-8")
        text2 = file2.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text1 = file1.read_text(encoding="cp1251")
        text2 = file2.read_text(encoding="cp1251")

    analyzer = create_analyzer(alphabet)
    comparison = analyzer.compare(text1, text2, file1.name, file2.name)

    console.print()
    console.print(f"[bold blue]🔄 Comparison:[/bold blue] {file1.name} vs {file2.name}")
    console.print()

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("KL Divergence (P||Q)", f"{comparison.kl_divergence_p_q:.4f}")
    table.add_row("KL Divergence (Q||P)", f"{comparison.kl_divergence_q_p:.4f}")
    table.add_row("Jensen-Shannon Divergence", f"{comparison.js_divergence:.4f}")
    if comparison.cosine_similarity:
        table.add_row("Cosine Similarity", f"{comparison.cosine_similarity:.4f}")

    console.print(table)


@app.command("split-analyze")
def split_analyze(
    file: Path | None = typer.Argument(None, help="Text file with segments (or use --text)"),  # noqa: B008
    text: str | None = typer.Option(None, "--text", "-t", help="Text content directly"),
    delimiter: str = typer.Option("***", "--delimiter", "-d", help="Delimiter to split segments"),
    alphabet: str = typer.Option("rus29", "--alphabet", "-a", help="Alphabet type"),
    output_html: Path | None = typer.Option(None, "--html", "-o", help="Output HTML chart"),  # noqa: B008
    output_csv: Path | None = typer.Option(None, "--csv", "-c", help="Output CSV file"),  # noqa: B008
    output_json: Path | None = typer.Option(None, "--json", "-j", help="Output JSON file"),  # noqa: B008
    show_chart: bool = typer.Option(False, "--show", "-s", help="Open chart in browser"),
):
    """
    Analyze text split by delimiters (e.g., ***).

    This command splits text by a delimiter and analyzes each segment,
    then shows entropy vs word count with linear regression.

    Examples:
        entropy-analysis split-analyze file.txt
        entropy-analysis split-analyze file.txt --delimiter="---"
        echo "text1 *** text2 *** text3" | entropy-analysis split-analyze --text -
    """
    # Get text content
    if text == "-":
        # Read from stdin
        import sys

        text = sys.stdin.read()
    elif file is not None:
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file.read_text(encoding="cp1251")
    elif text is None:
        console.print("[red]Error:[/red] Provide either a file or --text")
        raise typer.Exit(1)

    # Create analyzer
    analyzer = create_analyzer(alphabet)

    # Analyze
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing segments...", total=None)
        batch_result = analyzer.split_and_analyze(text, delimiter=delimiter)

    n_segments = len(batch_result.results)

    if n_segments == 0:
        console.print(f"[yellow]Warning:[/yellow] No segments found with delimiter '{delimiter}'")
        raise typer.Exit(0)

    console.print()
    console.print(f"[bold blue]📊 Split Analysis:[/bold blue] {n_segments} segments found")
    console.print(f"[dim]Delimiter: '{delimiter}'[/dim]")
    console.print()

    # Summary statistics
    if batch_result.extended_stats:
        stats = batch_result.extended_stats
        summary_table = Table(title="Entropy Statistics", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Mean H", f"{stats.mean:.4f} bits")
        summary_table.add_row("Median H", f"{stats.median:.4f} bits")
        summary_table.add_row("Std Dev", f"{stats.std_dev:.4f}")
        summary_table.add_row("Range", f"[{stats.min_value:.4f}, {stats.max_value:.4f}]")

        console.print(summary_table)
        console.print()

    # Correlation
    if batch_result.correlation is not None:
        corr_table = Table(title="Correlation: H vs N", show_header=True)
        corr_table.add_column("Metric", style="cyan")
        corr_table.add_column("Value", style="green")

        corr_table.add_row("Correlation (r)", f"{batch_result.correlation:.4f}")
        corr_table.add_row("R²", f"{batch_result.correlation_r_squared:.4f}")
        corr_table.add_row("p-value", f"{batch_result.correlation_p_value:.6f}")
        corr_table.add_row(
            "Equation",
            f"H = {batch_result.correlation_slope:.6f} × N + {batch_result.correlation_intercept:.4f}",
        )

        console.print(corr_table)
        console.print()

    # Segments table
    segments_table = Table(title="Segments", show_header=True)
    segments_table.add_column("#", justify="right", style="dim")
    segments_table.add_column("Name", style="cyan")
    segments_table.add_column("Words (N)", justify="right")
    segments_table.add_column("Entropy (H)", justify="right")
    segments_table.add_column("Mean Rank", justify="right")

    for i, (name, result) in enumerate(batch_result.results, 1):
        segments_table.add_row(
            str(i),
            name,
            f"{result.n_words:,}",
            f"{result.shannon_entropy:.4f}" if result.shannon_entropy else "—",
            f"{result.mean_rank:.2f}" if result.mean_rank else "—",
        )

    console.print(segments_table)

    # Create visualization
    if output_html or show_chart:
        from entropy_analysis.visualization import create_correlation_scatter

        fig = create_correlation_scatter(
            batch_result,
            show_trendline=True,
            highlight_outliers=True,
        )

        # Update title for split analysis
        fig.update_layout(
            title=f"Энтропия vs Количество слов<br><sub>{n_segments} сегментов, разделитель: '{delimiter}'</sub>"
        )

        if output_html:
            fig.write_html(str(output_html))
            console.print(f"\n[green]✓ Chart saved to:[/green] {output_html}")

        if show_chart:
            import tempfile
            import webbrowser

            if output_html:
                webbrowser.open(f"file://{output_html.absolute()}")
            else:
                # Create temp file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                    fig.write_html(f.name)
                    webbrowser.open(f"file://{f.name}")
                    console.print("\n[green]✓ Chart opened in browser[/green]")

    # Output CSV
    if output_csv:
        df = analyzer.batch_to_dataframe(batch_result)
        df.write_csv(output_csv)
        console.print(f"\n[green]✓ CSV saved to:[/green] {output_csv}")

    # Output JSON
    if output_json:
        data = {
            "delimiter": delimiter,
            "n_segments": n_segments,
            "correlation": batch_result.correlation,
            "r_squared": batch_result.correlation_r_squared,
            "p_value": batch_result.correlation_p_value,
            "slope": batch_result.correlation_slope,
            "intercept": batch_result.correlation_intercept,
            "segments": [
                {
                    "name": name,
                    "n_words": r.n_words,
                    "entropy": r.shannon_entropy,
                    "mean_rank": r.mean_rank,
                    "std_rank": r.std_rank,
                }
                for name, r in batch_result.results
            ],
        }
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        console.print(f"[green]✓ JSON saved to:[/green] {output_json}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
):
    """Start the FastAPI server."""
    from entropy_analysis.api.main import run_server

    console.print(f"[bold green]🚀 Starting API server at http://{host}:{port}[/bold green]")
    console.print(f"   📚 Documentation: http://{host}:{port}/docs")
    run_server(host=host, port=port)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit"),
):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    console.print(f"[bold green]🎨 Starting dashboard at http://localhost:{port}[/bold green]")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
    )


@app.command()
def version():
    """Show version information."""
    from entropy_analysis import __version__

    console.print(f"[bold]entropy-analysis[/bold] version {__version__}")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
