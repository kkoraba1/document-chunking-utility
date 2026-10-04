"""Tests for run-level summary reporting."""

from pathlib import Path

from document_chunking_utility.chunkers import (
    chunk_by_bounded_paragraph,
    chunk_by_fixed_words,
    chunk_by_paragraph,
)
from document_chunking_utility.io import SourceDocument
from document_chunking_utility.metadata import (
    ChunkRecord,
    build_chunk_records,
)
from document_chunking_utility.report import (
    RunSummary,
    build_run_summary,
    format_run_summary,
)

FIXTR_DIR = Path(__file__).parent / "fixtures" / "chunker_inputs"


def read_sample(file_name: str) -> str:
    """Read a fixture document without changing line endings."""
    file_path = FIXTR_DIR / file_name

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as sample_file:
        return sample_file.read()


def make_document(
    file_name: str,
    source_path: str | None = None,
) -> SourceDocument:
    """Build a source document from a fixture file."""
    return SourceDocument(
        source_file=file_name,
        source_path=source_path or file_name,
        text=read_sample(file_name),
    )


def make_para_records() -> list[ChunkRecord]:
    """Create paragraph records from a fixture."""
    src_doc = make_document("car_engines_essay.txt")
    chnk_list = chunk_by_paragraph(src_doc.text)

    return build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )


def make_fixed_records() -> list[ChunkRecord]:
    """Create fixed-size records from a fixture."""
    src_doc = make_document("bird_watching_article.txt")

    chnk_list = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=12,
        overlap=3,
    )

    return build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="fixed",
        chunk_size=12,
        overlap=3,
    )


def make_bound_records() -> list[ChunkRecord]:
    """Create bounded paragraph records from a fixture."""
    src_doc = make_document("car_engines_essay.txt")

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=20,
        tokenizer_name="cl100k_base",
    )

    return build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=20,
        tokenizer="cl100k_base",
    )


def grab_word_counts(
    rec_list: list[ChunkRecord],
) -> list[int]:
    """Return word counts from chunk records."""
    cnt_list = []

    for rec_item in rec_list:
        cnt_list.append(rec_item.word_count)

    return cnt_list


def grab_token_counts(
    rec_list: list[ChunkRecord],
) -> list[int]:
    """Return available token counts from chunk records."""
    cnt_list = []

    for rec_item in rec_list:
        if rec_item.token_count is not None:
            cnt_list.append(rec_item.token_count)

    return cnt_list


# -----------------------------------------------------------------------------
# Summary calculation
# -----------------------------------------------------------------------------


def test_build_run_summary_computes_mapping_statistics() -> None:
    rec_list = [
        {
            "chunk_id": "meeting_notes_000",
            "word_count": 12,
        },
        {
            "chunk_id": "meeting_notes_001",
            "word_count": 18,
        },
        {
            "chunk_id": "meeting_notes_002",
            "word_count": 30,
        },
    ]

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=2,
        unsupported_files_skipped=1,
        empty_files_skipped=1,
        read_error_files_skipped=1,
        output_file=r"outputs\reports\meeting_chunks.csv",
    )

    assert run_sum == RunSummary(
        processed_files=2,
        unsupported_files_skipped=1,
        empty_files_skipped=1,
        read_error_files_skipped=1,
        total_chunks=3,
        average_words_per_chunk=20.0,
        shortest_chunk_words=12,
        longest_chunk_words=30,
        output_file="outputs/reports/meeting_chunks.csv",
    )


def test_build_run_summary_accepts_chunk_records() -> None:
    rec_list = make_para_records()
    word_cnts = grab_word_counts(rec_list)

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file=Path("outputs/car_engine_chunks.csv"),
    )

    assert run_sum.total_chunks == 3
    assert run_sum.average_words_per_chunk == (
        sum(word_cnts) / len(word_cnts)
    )
    assert run_sum.shortest_chunk_words == min(word_cnts)
    assert run_sum.longest_chunk_words == max(word_cnts)
    assert run_sum.output_file == (
        "outputs/car_engine_chunks.csv"
    )


def test_build_run_summary_tracks_fixed_chunks() -> None:
    rec_list = make_fixed_records()
    word_cnts = grab_word_counts(rec_list)

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file="outputs/fixed_chunks.jsonl",
    )

    assert run_sum.total_chunks == 10
    assert run_sum.average_words_per_chunk == (
        sum(word_cnts) / len(word_cnts)
    )
    assert run_sum.shortest_chunk_words == min(word_cnts)
    assert run_sum.longest_chunk_words == max(word_cnts)

    assert run_sum.tokenized_chunks == 0
    assert run_sum.average_tokens_per_chunk == 0.0
    assert run_sum.shortest_chunk_tokens == 0
    assert run_sum.longest_chunk_tokens == 0


def test_build_run_summary_tracks_bounded_token_counts() -> None:
    rec_list = make_bound_records()
    word_cnts = grab_word_counts(rec_list)
    tok_cnts = grab_token_counts(rec_list)

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file="outputs/bounded_chunks.jsonl",
    )

    assert run_sum.total_chunks == len(rec_list)
    assert run_sum.average_words_per_chunk == (
        sum(word_cnts) / len(word_cnts)
    )
    assert run_sum.shortest_chunk_words == min(word_cnts)
    assert run_sum.longest_chunk_words == max(word_cnts)

    assert run_sum.tokenized_chunks == len(rec_list)
    assert run_sum.average_tokens_per_chunk == (
        sum(tok_cnts) / len(tok_cnts)
    )
    assert run_sum.shortest_chunk_tokens == min(tok_cnts)
    assert run_sum.longest_chunk_tokens == max(tok_cnts)
    assert run_sum.longest_chunk_tokens <= 20


def test_build_run_summary_reads_mapping_token_counts() -> None:
    rec_list = [
        {
            "word_count": 8,
            "token_count": 10,
        },
        {
            "word_count": 12,
            "token_count": 15,
        },
        {
            "word_count": 16,
            "token_count": 20,
        },
    ]

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file="outputs/bounded_chunks.csv",
    )

    assert run_sum.tokenized_chunks == 3
    assert run_sum.average_tokens_per_chunk == 15.0
    assert run_sum.shortest_chunk_tokens == 10
    assert run_sum.longest_chunk_tokens == 20


def test_build_run_summary_ignores_empty_token_counts() -> None:
    rec_list = [
        {
            "word_count": 8,
            "token_count": None,
        },
        {
            "word_count": 12,
            "token_count": "",
        },
        {
            "word_count": 16,
            "token_count": 20,
        },
    ]

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file="outputs/mixed_chunks.csv",
    )

    assert run_sum.tokenized_chunks == 1
    assert run_sum.average_tokens_per_chunk == 20.0
    assert run_sum.shortest_chunk_tokens == 20
    assert run_sum.longest_chunk_tokens == 20


def test_build_run_summary_normalizes_windows_path() -> None:
    run_sum = build_run_summary(
        records=[],
        processed_files=0,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file=r"outputs\nested\chunks.csv",
    )

    assert run_sum.output_file == (
        "outputs/nested/chunks.csv"
    )


def test_build_run_summary_handles_zero_chunks() -> None:
    run_sum = build_run_summary(
        records=[],
        processed_files=0,
        unsupported_files_skipped=2,
        empty_files_skipped=1,
        read_error_files_skipped=1,
        output_file="outputs/empty_chunks.csv",
    )

    assert run_sum.processed_files == 0
    assert run_sum.unsupported_files_skipped == 2
    assert run_sum.empty_files_skipped == 1
    assert run_sum.read_error_files_skipped == 1

    assert run_sum.total_chunks == 0
    assert run_sum.average_words_per_chunk == 0.0
    assert run_sum.shortest_chunk_words == 0
    assert run_sum.longest_chunk_words == 0

    assert run_sum.tokenized_chunks == 0
    assert run_sum.average_tokens_per_chunk == 0.0
    assert run_sum.shortest_chunk_tokens == 0
    assert run_sum.longest_chunk_tokens == 0


def test_build_run_summary_treats_missing_words_as_zero() -> None:
    rec_list = [
        {
            "chunk_id": "rough_note_000",
            "word_count": 14,
        },
        {
            "chunk_id": "rough_note_001",
        },
        {
            "chunk_id": "rough_note_002",
            "word_count": 7,
        },
    ]

    run_sum = build_run_summary(
        records=rec_list,
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        output_file="outputs/rough_chunks.csv",
    )

    assert run_sum.total_chunks == 3
    assert run_sum.average_words_per_chunk == 7.0
    assert run_sum.shortest_chunk_words == 0
    assert run_sum.longest_chunk_words == 14


# -----------------------------------------------------------------------------
# Summary formatting
# -----------------------------------------------------------------------------


def test_format_run_summary_contains_word_statistics() -> None:
    run_sum = RunSummary(
        processed_files=2,
        unsupported_files_skipped=1,
        empty_files_skipped=1,
        read_error_files_skipped=0,
        total_chunks=3,
        average_words_per_chunk=18.666666,
        shortest_chunk_words=4,
        longest_chunk_words=31,
        output_file="outputs/paragraph_chunks.csv",
    )

    sum_text = format_run_summary(run_sum)

    assert sum_text == (
        "Processed files: 2\n"
        "Unsupported files skipped: 1\n"
        "Empty files skipped: 1\n"
        "Total chunks: 3\n"
        "Average words per chunk: 18.7\n"
        "Shortest chunk: 4 words\n"
        "Longest chunk: 31 words\n"
        "Output file: outputs/paragraph_chunks.csv"
    )


def test_format_run_summary_contains_token_statistics() -> None:
    run_sum = RunSummary(
        processed_files=2,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        total_chunks=18,
        average_words_per_chunk=9.9,
        shortest_chunk_words=1,
        longest_chunk_words=18,
        output_file="outputs/bounded_chunks.jsonl",
        tokenized_chunks=18,
        average_tokens_per_chunk=11.4444,
        shortest_chunk_tokens=1,
        longest_chunk_tokens=20,
    )

    sum_text = format_run_summary(run_sum)

    assert sum_text == (
        "Processed files: 2\n"
        "Unsupported files skipped: 0\n"
        "Empty files skipped: 0\n"
        "Total chunks: 18\n"
        "Average words per chunk: 9.9\n"
        "Shortest chunk: 1 word\n"
        "Longest chunk: 18 words\n"
        "Tokenized chunks: 18\n"
        "Average tokens per chunk: 11.4\n"
        "Shortest token count: 1 token\n"
        "Longest token count: 20 tokens\n"
        "Output file: outputs/bounded_chunks.jsonl"
    )


def test_format_run_summary_includes_read_errors() -> None:
    run_sum = RunSummary(
        processed_files=1,
        unsupported_files_skipped=1,
        empty_files_skipped=1,
        read_error_files_skipped=2,
        total_chunks=4,
        average_words_per_chunk=11.25,
        shortest_chunk_words=5,
        longest_chunk_words=19,
        output_file="outputs/fixed_chunks.jsonl",
    )

    sum_text = format_run_summary(run_sum)

    assert "Read error files skipped: 2" in sum_text
    assert "Total chunks: 4" in sum_text


def test_format_run_summary_omits_read_errors_when_zero() -> None:
    run_sum = RunSummary(
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        total_chunks=3,
        average_words_per_chunk=24.0,
        shortest_chunk_words=18,
        longest_chunk_words=30,
        output_file="outputs/paragraph_chunks.csv",
    )

    sum_text = format_run_summary(run_sum)

    assert "Read error files skipped:" not in sum_text


def test_format_run_summary_omits_tokens_when_unavailable() -> None:
    run_sum = RunSummary(
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        total_chunks=3,
        average_words_per_chunk=24.0,
        shortest_chunk_words=18,
        longest_chunk_words=30,
        output_file="outputs/paragraph_chunks.csv",
    )

    sum_text = format_run_summary(run_sum)

    assert "Tokenized chunks:" not in sum_text
    assert "Average tokens per chunk:" not in sum_text
    assert "Shortest token count:" not in sum_text
    assert "Longest token count:" not in sum_text


def test_format_run_summary_handles_zero_chunks() -> None:
    run_sum = RunSummary(
        processed_files=0,
        unsupported_files_skipped=1,
        empty_files_skipped=1,
        total_chunks=0,
        average_words_per_chunk=0.0,
        shortest_chunk_words=0,
        longest_chunk_words=0,
        output_file="outputs/empty_chunks.csv",
    )

    sum_text = format_run_summary(run_sum)

    assert "Processed files: 0" in sum_text
    assert "Total chunks: 0" in sum_text
    assert "Average words per chunk: 0.0" in sum_text
    assert "Shortest chunk: 0 words" in sum_text
    assert "Longest chunk: 0 words" in sum_text
    assert "Tokenized chunks:" not in sum_text


def test_format_run_summary_rounds_averages() -> None:
    run_sum = RunSummary(
        processed_files=1,
        unsupported_files_skipped=0,
        empty_files_skipped=0,
        total_chunks=3,
        average_words_per_chunk=13.333333,
        shortest_chunk_words=9,
        longest_chunk_words=18,
        output_file="outputs/rounded_summary.csv",
        tokenized_chunks=3,
        average_tokens_per_chunk=17.666666,
        shortest_chunk_tokens=12,
        longest_chunk_tokens=20,
    )

    sum_text = format_run_summary(run_sum)

    assert "Average words per chunk: 13.3" in sum_text
    assert "Average tokens per chunk: 17.7" in sum_text