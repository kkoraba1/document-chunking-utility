"""Run-level reporting helpers for the Document Chunking Utility."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_chunking_utility.paths import to_portable_path


@dataclass(frozen=True)
class RunSummary:
    processed_files: int
    unsupported_files_skipped: int
    empty_files_skipped: int
    total_chunks: int
    average_words_per_chunk: float
    shortest_chunk_words: int
    longest_chunk_words: int
    output_file: str
    read_error_files_skipped: int = 0
    tokenized_chunks: int = 0
    average_tokens_per_chunk: float = 0.0
    shortest_chunk_tokens: int = 0
    longest_chunk_tokens: int = 0


def build_run_summary(
    records: Sequence[Mapping[str, Any] | object],
    processed_files: int,
    unsupported_files_skipped: int,
    empty_files_skipped: int,
    output_file: str | Path,
    read_error_files_skipped: int = 0,
) -> RunSummary:
    """Build summary statistics for a chunking run."""
    word_counts: list[int] = []
    token_counts: list[int] = []

    # Collect the word and token counts from each chunk record.
    for record in records:
        word_count = _get_word_count(record)
        word_counts.append(word_count)

        token_count = _get_token_count(record)

        if token_count is not None:
            token_counts.append(token_count)

    total_chunks = len(word_counts)

    if total_chunks == 0:
        avg_words = 0.0
        shortest_words = 0
        longest_words = 0
    else:
        avg_words = sum(word_counts) / total_chunks
        shortest_words = min(word_counts)
        longest_words = max(word_counts)

    tokenized_chunks = len(token_counts)

    if tokenized_chunks == 0:
        avg_tokens = 0.0
        shortest_tokens = 0
        longest_tokens = 0
    else:
        avg_tokens = sum(token_counts) / tokenized_chunks
        shortest_tokens = min(token_counts)
        longest_tokens = max(token_counts)

    return RunSummary(
        processed_files=processed_files,
        unsupported_files_skipped=unsupported_files_skipped,
        empty_files_skipped=empty_files_skipped,
        total_chunks=total_chunks,
        average_words_per_chunk=avg_words,
        shortest_chunk_words=shortest_words,
        longest_chunk_words=longest_words,
        output_file=to_portable_path(output_file),
        read_error_files_skipped=read_error_files_skipped,
        tokenized_chunks=tokenized_chunks,
        average_tokens_per_chunk=avg_tokens,
        shortest_chunk_tokens=shortest_tokens,
        longest_chunk_tokens=longest_tokens,
    )


def format_run_summary(summary: RunSummary) -> str:
    """Format run statistics for command-line output."""
    summary_lines = [
        f"Processed files: {summary.processed_files}",
        (
            "Unsupported files skipped: "
            f"{summary.unsupported_files_skipped}"
        ),
        f"Empty files skipped: {summary.empty_files_skipped}",
    ]

    if summary.read_error_files_skipped:
        summary_lines.append(
            "Read error files skipped: "
            f"{summary.read_error_files_skipped}"
        )

    summary_lines.extend(
        [
            f"Total chunks: {summary.total_chunks}",
            (
                "Average words per chunk: "
                f"{summary.average_words_per_chunk:.1f}"
            ),
            (
                "Shortest chunk: "
                f"{format_count(summary.shortest_chunk_words, 'word')}"
            ),
            (
                "Longest chunk: "
                f"{format_count(summary.longest_chunk_words, 'word')}"
            ),
        ]
    )

    # Only show token statistics when token-aware chunks were used.
    if summary.tokenized_chunks > 0:
        summary_lines.extend(
            [
                f"Tokenized chunks: {summary.tokenized_chunks}",
                (
                    "Average tokens per chunk: "
                    f"{summary.average_tokens_per_chunk:.1f}"
                ),
                (
                    "Shortest token count: "
                    f"{format_count(summary.shortest_chunk_tokens, 'token')}"
                ),
                (
                    "Longest token count: "
                    f"{format_count(summary.longest_chunk_tokens, 'token')}"
                ),
            ]
        )

    summary_lines.append(
        f"Output file: {summary.output_file}"
    )

    return "\n".join(summary_lines)


def format_count(count: int, unit: str) -> str:
    """Format a count using the correct singular or plural label."""
    unit_label = unit

    if count != 1:
        unit_label = unit + "s"

    return f"{count} {unit_label}"


def _get_word_count(
    record: Mapping[str, Any] | object,
) -> int:
    """Read the word count from a mapping or record object."""
    if isinstance(record, Mapping):
        value = record.get("word_count", 0)
    else:
        value = getattr(record, "word_count", 0)

    return int(value)


def _get_token_count(
    record: Mapping[str, Any] | object,
) -> int | None:
    """Read the optional token count from a mapping or record object."""
    if isinstance(record, Mapping):
        value = record.get("token_count")
    else:
        value = getattr(record, "token_count", None)

    if value is None or value == "":
        return None

    return int(value)