"""Tests for document loading and output writing helpers."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from document_chunking_utility.io import (
    OUTPUT_FIELDNAMES,
    LoadResult,
    SourceDocument,
    discover_files,
    load_documents,
    read_text_file,
    write_csv,
    write_jsonl,
    write_records,
)
from document_chunking_utility.metadata import ChunkRecord
from document_chunking_utility.validation import ValidationError

FIXTR_DIR = Path(__file__).parent / "fixtures" / "chunker_inputs"

EXPECT_FIELDS = [
    "chunk_id",
    "source_file",
    "source_path",
    "method",
    "chunk_size",
    "overlap",
    "token_count",
    "max_tokens",
    "tokenizer",
    "chunk_index",
    "chunk_text",
    "start_char",
    "end_char",
    "char_count",
    "word_count",
]


def copy_sample(
    docs_dir: Path,
    file_name: str,
    target_name: str | None = None,
) -> Path:
    """Copy a fixture document into a temporary folder."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    dest_name = file_name

    if target_name is not None:
        dest_name = target_name

    dest_path = docs_dir / dest_name

    shutil.copyfile(
        FIXTR_DIR / file_name,
        dest_path,
    )

    return dest_path


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    """Read CSV output rows as dictionaries."""
    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def read_jsonl_rows(file_path: Path) -> list[dict[str, object]]:
    """Read JSONL output records as dictionaries."""
    rec_list = []

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec_list.append(json.loads(line))

    return rec_list


def sample_records() -> list[dict[str, object]]:
    """Return sample paragraph chunk records."""
    return [
        {
            "chunk_id": "course_notes_000",
            "source_file": "course_notes.txt",
            "source_path": "semester/course_notes.txt",
            "method": "paragraph",
            "chunk_size": None,
            "overlap": None,
            "token_count": None,
            "max_tokens": None,
            "tokenizer": None,
            "chunk_index": 0,
            "chunk_text": (
                "Arrays store items next to each other in memory."
            ),
            "start_char": 0,
            "end_char": 47,
            "char_count": 47,
            "word_count": 8,
        },
        {
            "chunk_id": "course_notes_001",
            "source_file": "course_notes.txt",
            "source_path": "semester/course_notes.txt",
            "method": "paragraph",
            "chunk_size": None,
            "overlap": None,
            "token_count": None,
            "max_tokens": None,
            "tokenizer": None,
            "chunk_index": 1,
            "chunk_text": (
                "Linked lists are easier to grow, "
                "but searching takes longer."
            ),
            "start_char": 49,
            "end_char": 108,
            "char_count": 59,
            "word_count": 10,
        },
    ]


def bounded_record() -> dict[str, object]:
    """Return one bounded paragraph chunk record."""
    return {
        "chunk_id": "bounded_notes_000",
        "source_file": "bounded_notes.txt",
        "source_path": "notes/bounded_notes.txt",
        "method": "bounded-paragraph",
        "chunk_size": None,
        "overlap": None,
        "token_count": 12,
        "max_tokens": 20,
        "tokenizer": "cl100k_base",
        "chunk_index": 0,
        "chunk_text": "A bounded paragraph chunk with token metadata.",
        "start_char": 0,
        "end_char": 46,
        "char_count": 46,
        "word_count": 7,
    }


# -----------------------------------------------------------------------------
# File discovery and reading
# -----------------------------------------------------------------------------


def test_discover_files_returns_files_in_stable_order(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"
    week_dir = docs_dir / "week_02"

    copy_sample(
        docs_dir,
        "bird_watching_article.txt",
        "z_bird_notes.txt",
    )
    copy_sample(
        docs_dir,
        "car_engines_essay.txt",
        "a_engine_essay.txt",
    )
    copy_sample(
        archive_dir,
        "construction_meeting_notes.md",
        "meeting_notes.md",
    )
    copy_sample(
        week_dir,
        "data_structures_lecture_note.txt",
        "lecture_note.txt",
    )

    file_list = discover_files(docs_dir)
    rel_paths = []

    for file_path in file_list:
        rel_paths.append(
            file_path.relative_to(docs_dir).as_posix()
        )

    assert rel_paths == [
        "a_engine_essay.txt",
        "archive/meeting_notes.md",
        "week_02/lecture_note.txt",
        "z_bird_notes.txt",
    ]


def test_discover_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing_folder"

    with pytest.raises(
        ValidationError,
        match="Input path does not exist",
    ):
        discover_files(missing_dir)


def test_discover_files_rejects_file_instead_of_directory(
    tmp_path: Path,
) -> None:
    file_path = copy_sample(
        tmp_path,
        "data_structures_lecture_note.txt",
    )

    with pytest.raises(
        ValidationError,
        match="Input path is not a directory",
    ):
        discover_files(file_path)


def test_read_text_file_reads_utf8_fixture_text(
    tmp_path: Path,
) -> None:
    file_path = copy_sample(
        tmp_path,
        "unicode_japanese_notes.txt",
    )

    src_text = read_text_file(file_path)

    assert "東京" in src_text
    assert "Δx" in src_text
    assert "🙂" in src_text


def test_read_text_file_preserves_crlf_line_endings(
    tmp_path: Path,
) -> None:
    file_path = copy_sample(
        tmp_path,
        "professor_office_hours_windows.txt",
    )
    expected_text = file_path.read_bytes().decode("utf-8")

    src_text = read_text_file(file_path)

    assert "\r\n" in src_text
    assert src_text == expected_text


def test_read_text_file_reports_invalid_utf8(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "bad_scan.txt"
    file_path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(
        ValidationError,
        match="Could not decode supported file as UTF-8",
    ):
        read_text_file(file_path)


# -----------------------------------------------------------------------------
# Document loading
# -----------------------------------------------------------------------------


def test_load_documents_loads_supported_documents(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"

    copy_sample(
        docs_dir,
        "car_engines_essay.txt",
    )
    copy_sample(
        docs_dir,
        "construction_meeting_notes.md",
    )

    load_res = load_documents(docs_dir)

    assert isinstance(load_res, LoadResult)
    assert load_res.processed_files == 2
    assert load_res.unsupported_files_skipped == 0
    assert load_res.empty_files_skipped == 0
    assert load_res.read_error_files_skipped == 0

    assert load_res.documents[0].source_file == (
        "car_engines_essay.txt"
    )
    assert load_res.documents[0].source_path == (
        "car_engines_essay.txt"
    )
    assert load_res.documents[0].text.startswith("Car Engines")

    assert load_res.documents[1].source_file == (
        "construction_meeting_notes.md"
    )
    assert load_res.documents[1].source_path == (
        "construction_meeting_notes.md"
    )
    assert load_res.documents[1].text.startswith(
        "# Construction Meeting Notes"
    )


def test_load_documents_skips_unusable_files(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"

    copy_sample(
        docs_dir,
        "data_structures_lecture_note.txt",
    )
    copy_sample(
        docs_dir,
        "blank_document.txt",
    )

    (docs_dir / "diagram.png").write_text(
        "pretend image data",
        encoding="utf-8",
    )
    (docs_dir / "bad_export.txt").write_bytes(
        b"\xff\xfe\x00\x00"
    )

    load_res = load_documents(docs_dir)

    assert load_res.processed_files == 1
    assert load_res.unsupported_files_skipped == 1
    assert load_res.empty_files_skipped == 1
    assert load_res.read_error_files_skipped == 1
    assert len(load_res.documents) == 1
    assert load_res.documents[0].source_file == (
        "data_structures_lecture_note.txt"
    )


def test_load_documents_preserves_nested_source_paths(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "semester_one" / "week_03"

    copy_sample(
        nested_dir,
        "data_structures_lecture_note.txt",
        "lecture_note.txt",
    )

    load_res = load_documents(docs_dir)

    assert load_res.documents == [
        SourceDocument(
            source_file="lecture_note.txt",
            source_path=(
                "semester_one/week_03/lecture_note.txt"
            ),
            text=(nested_dir / "lecture_note.txt").read_text(
                encoding="utf-8"
            ),
        )
    ]


def test_load_documents_returns_empty_result_without_usable_files(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    copy_sample(
        docs_dir,
        "blank_document.txt",
    )

    (docs_dir / "spreadsheet.csv").write_text(
        "name,value\nrow,1",
        encoding="utf-8",
    )

    load_res = load_documents(docs_dir)

    assert load_res.documents == []
    assert load_res.processed_files == 0
    assert load_res.unsupported_files_skipped == 1
    assert load_res.empty_files_skipped == 1
    assert load_res.read_error_files_skipped == 0


# -----------------------------------------------------------------------------
# Output schema
# -----------------------------------------------------------------------------


def test_output_fieldnames_match_documented_schema() -> None:
    assert OUTPUT_FIELDNAMES == EXPECT_FIELDS


def test_write_csv_writes_schema_and_records(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "exports" / "chunks.csv"

    result_path = write_csv(
        sample_records(),
        out_path,
    )

    assert result_path == out_path
    assert out_path.exists()

    row_list = read_csv_rows(out_path)

    assert list(row_list[0].keys()) == EXPECT_FIELDS
    assert len(row_list) == 2
    assert row_list[0]["chunk_id"] == "course_notes_000"
    assert row_list[0]["source_path"] == (
        "semester/course_notes.txt"
    )
    assert row_list[0]["chunk_size"] == ""
    assert row_list[0]["overlap"] == ""
    assert row_list[0]["token_count"] == ""
    assert row_list[0]["max_tokens"] == ""
    assert row_list[0]["tokenizer"] == ""
    assert row_list[1]["chunk_index"] == "1"
    assert row_list[1]["word_count"] == "10"


def test_write_jsonl_writes_schema_and_records(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "exports" / "chunks.jsonl"

    result_path = write_jsonl(
        sample_records(),
        out_path,
    )

    assert result_path == out_path
    assert out_path.exists()

    rec_list = read_jsonl_rows(out_path)

    assert list(rec_list[0].keys()) == EXPECT_FIELDS
    assert len(rec_list) == 2
    assert rec_list[0]["chunk_id"] == "course_notes_000"
    assert rec_list[0]["chunk_text"].startswith(
        "Arrays store"
    )
    assert rec_list[0]["chunk_size"] is None
    assert rec_list[0]["overlap"] is None
    assert rec_list[0]["token_count"] is None
    assert rec_list[0]["max_tokens"] is None
    assert rec_list[0]["tokenizer"] is None
    assert rec_list[1]["chunk_index"] == 1


def test_write_csv_preserves_bounded_token_metadata(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "bounded_chunks.csv"

    write_csv(
        [bounded_record()],
        out_path,
    )

    row_list = read_csv_rows(out_path)
    row_item = row_list[0]

    assert row_item["method"] == "bounded-paragraph"
    assert row_item["token_count"] == "12"
    assert row_item["max_tokens"] == "20"
    assert row_item["tokenizer"] == "cl100k_base"


def test_write_jsonl_preserves_bounded_token_metadata(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "bounded_chunks.jsonl"

    write_jsonl(
        [bounded_record()],
        out_path,
    )

    rec_list = read_jsonl_rows(out_path)
    rec_item = rec_list[0]

    assert rec_item["method"] == "bounded-paragraph"
    assert rec_item["token_count"] == 12
    assert rec_item["max_tokens"] == 20
    assert rec_item["tokenizer"] == "cl100k_base"


def test_write_jsonl_preserves_unicode(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "unicode_chunks.jsonl"

    rec_list = sample_records()
    rec_list[0]["chunk_id"] = "field_note_000"
    rec_list[0]["chunk_text"] = (
        "東京 station note with Δx and a 🙂 mark."
    )
    rec_list[0]["char_count"] = 39
    rec_list[0]["word_count"] = 8

    write_jsonl(
        rec_list[:1],
        out_path,
    )

    raw_text = out_path.read_text(encoding="utf-8")
    parsed_list = read_jsonl_rows(out_path)

    assert "東京" in raw_text
    assert "\\u6771" not in raw_text
    assert parsed_list[0]["chunk_text"] == (
        "東京 station note with Δx and a 🙂 mark."
    )


def test_write_records_infers_csv_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "nested" / "chunks.csv"

    result_path = write_records(
        sample_records(),
        out_path,
    )

    assert result_path == out_path
    assert out_path.exists()
    assert read_csv_rows(out_path)[0]["chunk_id"] == (
        "course_notes_000"
    )


def test_write_records_infers_jsonl_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "nested" / "chunks.jsonl"

    result_path = write_records(
        sample_records(),
        out_path,
    )

    assert result_path == out_path
    assert out_path.exists()
    assert len(read_jsonl_rows(out_path)) == 2


def test_write_records_accepts_dataclass_records(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "dataclass_records.csv"

    rec_list = [
        ChunkRecord(
            chunk_id="lecture_note_000",
            source_file="lecture_note.txt",
            source_path="notes/lecture_note.txt",
            method="bounded-paragraph",
            chunk_size=None,
            overlap=None,
            token_count=10,
            max_tokens=20,
            tokenizer="cl100k_base",
            chunk_index=0,
            chunk_text=(
                "Stacks and queues restrict how items are removed."
            ),
            start_char=0,
            end_char=48,
            char_count=48,
            word_count=8,
        )
    ]

    write_records(
        rec_list,
        out_path,
    )

    row_list = read_csv_rows(out_path)

    assert row_list[0]["chunk_id"] == "lecture_note_000"
    assert row_list[0]["method"] == "bounded-paragraph"
    assert row_list[0]["token_count"] == "10"
    assert row_list[0]["max_tokens"] == "20"
    assert row_list[0]["tokenizer"] == "cl100k_base"


def test_write_records_writes_empty_csv_with_header(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "empty_chunks.csv"

    write_records(
        [],
        out_path,
    )

    row_list = read_csv_rows(out_path)
    raw_text = out_path.read_text(encoding="utf-8")

    assert row_list == []
    assert raw_text.startswith("chunk_id,")
    assert "token_count" in raw_text
    assert "max_tokens" in raw_text
    assert "tokenizer" in raw_text


def test_write_records_writes_empty_jsonl_file(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "empty_chunks.jsonl"

    write_records(
        [],
        out_path,
    )

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


def test_write_records_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.txt"

    with pytest.raises(
        ValidationError,
        match="Unsupported output file type",
    ):
        write_records(
            sample_records(),
            out_path,
        )


def test_write_csv_rejects_non_csv_path(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.jsonl"

    with pytest.raises(
        ValidationError,
        match="Output path must end with .csv",
    ):
        write_csv(
            sample_records(),
            out_path,
        )


def test_write_jsonl_rejects_non_jsonl_path(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.csv"

    with pytest.raises(
        ValidationError,
        match="Output path must end with .jsonl",
    ):
        write_jsonl(
            sample_records(),
            out_path,
        )


def test_write_records_rejects_plain_objects(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "bad_records.csv"

    with pytest.raises(
        TypeError,
        match="Chunk records must be dictionaries",
    ):
        write_records(
            [object()],
            out_path,
        )