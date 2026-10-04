import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from document_chunking_utility.chunkers import chunk_text
from document_chunking_utility.io import (
    load_documents,
    validate_output_path,
    write_records,
)
from document_chunking_utility.metadata import build_chunk_records
from document_chunking_utility.paths import to_portable_path
from document_chunking_utility.report import (
    build_run_summary,
    format_count,
    format_run_summary,
)
from document_chunking_utility.tokenizer import (
    DEFAULT_TOKENIZER,
    load_tokenizer,
)
from document_chunking_utility.validation import (
    ValidationError,
    validate_chunk_settings,
    validate_max_tokens,
)

app = typer.Typer(
    help="Split text and Markdown documents into metadata-rich chunks."
)


@app.command()
def chunk(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing .txt and .md source documents."
        ),
    ],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help=(
                "Chunking method: paragraph, bounded-paragraph, "
                "or fixed."
            ),
        ),
    ] = "paragraph",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output file path ending in .csv or .jsonl.",
        ),
    ] = Path("outputs/chunks.csv"),
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Number of words per fixed-size chunk.",
        ),
    ] = 200,
    overlap: Annotated[
        int,
        typer.Option(
            "--overlap",
            help="Number of shared words between fixed chunks.",
        ),
    ] = 40,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Maximum tokens per bounded paragraph chunk.",
        ),
    ] = 512,
    tokenizer: Annotated[
        str,
        typer.Option(
            "--tokenizer",
            help="Tokenizer used for bounded paragraph chunks.",
        ),
    ] = DEFAULT_TOKENIZER,
) -> None:
    """Chunk supported documents and write the results to a file."""
    try:
        norm_method = _normalize_method(method)
        norm_tokenizer = tokenizer.strip()

        # Only validate settings that belong to the selected method.
        if norm_method == "fixed":
            validate_chunk_settings(
                chunk_size=chunk_size,
                overlap=overlap,
            )

        if norm_method == "bounded-paragraph":
            validate_max_tokens(max_tokens)
            load_tokenizer(norm_tokenizer)

        load_result = load_documents(input_dir)
        chunk_records = []

        # Process each supported document one at a time.
        for document in load_result.documents:
            merge_md_heads = _should_merge_markdown_headings(
                source_file=document.source_file,
                method=norm_method,
            )

            doc_chunks = chunk_text(
                text=document.text,
                method=norm_method,
                chunk_size=chunk_size,
                overlap=overlap,
                max_tokens=max_tokens,
                tokenizer_name=norm_tokenizer,
                merge_markdown_headings=merge_md_heads,
            )

            doc_records = build_chunk_records(
                document=document,
                chunks=doc_chunks,
                method=norm_method,
                chunk_size=(
                    chunk_size
                    if norm_method == "fixed"
                    else None
                ),
                overlap=(
                    overlap
                    if norm_method == "fixed"
                    else None
                ),
                max_tokens=(
                    max_tokens
                    if norm_method == "bounded-paragraph"
                    else None
                ),
                tokenizer=(
                    norm_tokenizer
                    if norm_method == "bounded-paragraph"
                    else None
                ),
            )

            chunk_records.extend(doc_records)

        written_path = write_records(
            records=chunk_records,
            output_path=out,
        )

        run_summary = build_run_summary(
            records=chunk_records,
            processed_files=load_result.processed_files,
            unsupported_files_skipped=(
                load_result.unsupported_files_skipped
            ),
            empty_files_skipped=load_result.empty_files_skipped,
            read_error_files_skipped=(
                load_result.read_error_files_skipped
            ),
            output_file=written_path,
        )

    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Input directory: {input_dir}")
    typer.echo(f"Method: {norm_method}")

    if norm_method == "fixed":
        typer.echo(f"Chunk size: {chunk_size}")
        typer.echo(f"Overlap: {overlap}")

    if norm_method == "bounded-paragraph":
        typer.echo(f"Maximum tokens: {max_tokens}")
        typer.echo(f"Tokenizer: {norm_tokenizer}")

    typer.echo("")
    typer.echo(format_run_summary(run_summary))


@app.command()
def inspect(
    output_file: Annotated[
        Path,
        typer.Argument(
            help="CSV or JSONL chunk output file to inspect."
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Number of chunk previews to display.",
        ),
    ] = 5,
) -> None:
    """Inspect an exported CSV or JSONL chunk file."""
    try:
        records = _load_output_records(output_file)
        preview_limit = _validate_preview_limit(limit)

        inspect_data = _build_output_inspection(
            records=records,
            output_file=output_file,
        )

    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Output file: {inspect_data['output_file']}")
    typer.echo(f"Format: {inspect_data['format']}")
    typer.echo(f"Total chunks: {inspect_data['total_chunks']}")
    typer.echo(f"Source files: {inspect_data['source_files']}")
    typer.echo(f"Methods: {inspect_data['methods']}")

    typer.echo(
        "Average words per chunk: "
        f"{inspect_data['average_words_per_chunk']:.1f}"
    )
    typer.echo(
        "Shortest chunk: "
        f"{format_count(inspect_data['shortest_chunk_words'], 'word')}"
    )
    typer.echo(
        "Longest chunk: "
        f"{format_count(inspect_data['longest_chunk_words'], 'word')}"
    )

    if inspect_data["tokenized_chunks"] > 0:
        typer.echo(
            f"Tokenized chunks: {inspect_data['tokenized_chunks']}"
        )
        typer.echo(
            "Average tokens per chunk: "
            f"{inspect_data['average_tokens_per_chunk']:.1f}"
        )
        typer.echo(
            "Shortest token count: "
            f"{format_count(inspect_data['shortest_chunk_tokens'], 'token')}"
        )
        typer.echo(
            "Longest token count: "
            f"{format_count(inspect_data['longest_chunk_tokens'], 'token')}"
        )

    if records and preview_limit > 0:
        typer.echo("")
        typer.echo(
            f"Preview records: {min(preview_limit, len(records))}"
        )

        for record in records[:preview_limit]:
            typer.echo(_format_preview_record(record))


def _normalize_method(method: str) -> str:
    """Normalize and validate the selected chunking method."""
    norm_method = method.lower().strip()

    supported_methods = {
        "paragraph",
        "bounded-paragraph",
        "fixed",
    }

    if norm_method not in supported_methods:
        raise ValidationError(
            "Method must be 'paragraph', "
            "'bounded-paragraph', or 'fixed'."
        )

    return norm_method


def _should_merge_markdown_headings(
    source_file: str,
    method: str,
) -> bool:
    """Return True when Markdown headings should be merged."""
    file_ext = Path(source_file).suffix.lower()

    if file_ext != ".md":
        return False

    if method == "paragraph":
        return True

    return method == "bounded-paragraph"


def _load_output_records(
    output_file: str | Path,
) -> list[dict[str, Any]]:
    """Load chunk records from a supported output file."""
    output_path = validate_output_path(output_file)

    if not output_path.exists():
        raise ValidationError(
            f"Output file does not exist: {output_path}"
        )

    if not output_path.is_file():
        raise ValidationError(
            f"Output path is not a file: {output_path}"
        )

    if output_path.suffix.lower() == ".csv":
        return _read_csv_records(output_path)

    if output_path.suffix.lower() == ".jsonl":
        return _read_jsonl_records(output_path)

    raise ValidationError(
        f"Unsupported output file type: {output_path.suffix}"
    )


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Read all records from a CSV output file."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL output file."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_num, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Invalid JSONL at line {line_num}: "
                    f"{exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValidationError(
                    f"Invalid JSONL at line {line_num}: "
                    "expected object"
                )

            records.append(record)

    return records


def _validate_preview_limit(limit: int) -> int:
    """Make sure the preview limit is not negative."""
    if limit < 0:
        raise ValidationError(
            "Preview limit must be greater than or equal to 0."
        )

    return limit


def _build_output_inspection(
    records: list[Mapping[str, Any]],
    output_file: str | Path,
) -> dict[str, Any]:
    """Build summary information about an exported chunk file."""
    word_counts = []
    token_counts = []

    # Collect word and token counts from every record.
    for record in records:
        word_count = _safe_int(
            record.get("word_count", 0)
        )
        word_counts.append(word_count)

        token_count = _safe_optional_int(
            record.get("token_count")
        )

        if token_count is not None:
            token_counts.append(token_count)

    total_chunks = len(records)

    if total_chunks == 0:
        avg_words = 0.0
        min_words = 0
        max_words = 0

    else:
        avg_words = sum(word_counts) / total_chunks
        min_words = min(word_counts)
        max_words = max(word_counts)

    tokenized_chunks = len(token_counts)

    if tokenized_chunks == 0:
        avg_tokens = 0.0
        min_tokens = 0
        max_tokens = 0

    else:
        avg_tokens = sum(token_counts) / tokenized_chunks
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

    source_paths = set()
    used_methods = set()

    # Find the unique source files and chunking methods.
    for record in records:
        source_path = str(
            record.get("source_path")
            or record.get("source_file")
            or ""
        ).strip()

        if source_path:
            source_paths.add(source_path)

        record_method = str(
            record.get("method", "")
        ).strip()

        if record_method:
            used_methods.add(record_method)

    output_path = Path(output_file)

    if used_methods:
        method_text = ", ".join(sorted(used_methods))
    else:
        method_text = "<none>"

    return {
        "output_file": to_portable_path(output_file),
        "format": output_path.suffix.lower().lstrip(".").upper(),
        "total_chunks": total_chunks,
        "source_files": len(source_paths),
        "methods": method_text,
        "average_words_per_chunk": avg_words,
        "shortest_chunk_words": min_words,
        "longest_chunk_words": max_words,
        "tokenized_chunks": tokenized_chunks,
        "average_tokens_per_chunk": avg_tokens,
        "shortest_chunk_tokens": min_tokens,
        "longest_chunk_tokens": max_tokens,
    }


def _format_preview_record(
    record: Mapping[str, Any],
) -> str:
    """Format one chunk record for the inspection preview."""
    chunk_idx = record.get("chunk_index", "")
    chunk_id = record.get("chunk_id", "")
    source_file = record.get("source_file", "")

    word_count = _safe_int(
        record.get("word_count", 0)
    )
    token_count = _safe_optional_int(
        record.get("token_count")
    )

    preview_text = _truncate_preview(
        str(record.get("chunk_text", ""))
    )

    preview_line = (
        f"[{chunk_idx}] {chunk_id} | {source_file} | "
        f"{word_count} words"
    )

    if token_count is not None:
        preview_line = (
            f"{preview_line} | {token_count} tokens"
        )

    return f"{preview_line} | {preview_text}"


def _truncate_preview(
    text: str,
    max_chars: int = 80,
) -> str:
    """Shorten preview text when it is longer than the limit."""
    clean_text = " ".join(text.split())

    if len(clean_text) <= max_chars:
        return clean_text

    return f"{clean_text[: max_chars - 3]}..."


def _safe_int(value: Any) -> int:
    """Convert a value to an integer or return zero."""
    try:
        return int(value)

    except (TypeError, ValueError):
        return 0


def _safe_optional_int(value: Any) -> int | None:
    """Convert a value to an integer when one is available."""
    if value is None:
        return None

    if value == "":
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    app()