import csv
import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from document_chunking_utility.cli import app
from document_chunking_utility.tokenizer import count_tokens

cli_runner = CliRunner()

FIXTR_DIR = Path(__file__).parent / "fixtures" / "chunker_inputs"

EXPORT_FIELDS = [
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


def copy_fxtr(
    doc_dir: Path,
    file_name: str,
    target_name: str | None = None,
) -> Path:
    """Copy a fixture document into a temporary input folder."""
    doc_dir.mkdir(parents=True, exist_ok=True)

    dest_name = file_name

    if target_name is not None:
        dest_name = target_name

    dest_path = doc_dir / dest_name

    shutil.copyfile(FIXTR_DIR / file_name, dest_path)

    return dest_path


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read exported CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_jsonl_recs(path: Path) -> list[dict[str, object]]:
    """Read exported JSONL records as dictionaries."""
    rec_list = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec_list.append(json.loads(line))

    return rec_list


def split_words(text: str) -> list[str]:
    """Split text into whitespace-delimited words."""
    return text.split()


def assert_csv_layout(row_list: list[dict[str, str]]) -> None:
    """CSV exports should use the documented schema."""
    assert row_list
    assert list(row_list[0].keys()) == EXPORT_FIELDS


def assert_jsonl_layout(rec_list: list[dict[str, object]]) -> None:
    """JSONL exports should use the documented schema."""
    assert rec_list
    assert list(rec_list[0].keys()) == EXPORT_FIELDS


def assert_export_offsets_match(
    src_text: str,
    rec_list: list[dict],
) -> None:
    """Exported offsets should reconstruct every chunk."""
    for rec_item in rec_list:
        start_char = int(rec_item["start_char"])
        end_char = int(rec_item["end_char"])

        assert start_char < end_char

        src_slice = src_text[start_char:end_char]

        assert src_slice == rec_item["chunk_text"]
        assert int(rec_item["char_count"]) == len(rec_item["chunk_text"])
        assert int(rec_item["word_count"]) == len(rec_item["chunk_text"].split())


def assert_bounded_recs_contiguous(
    src_text: str,
    rec_list: list[dict],
) -> None:
    """Bounded paragraph exports should cover the source in order."""
    if not rec_list:
        assert src_text.strip() == ""
        return

    next_start = 0

    for rec_item in rec_list:
        start_char = int(rec_item["start_char"])
        end_char = int(rec_item["end_char"])

        assert start_char == next_start
        next_start = end_char

    assert next_start == len(src_text)


def assert_bounded_recs_rebuild_source(
    src_text: str,
    rec_list: list[dict],
) -> None:
    """Joining bounded paragraph export text should rebuild the source."""
    rebuilt_src = ""

    for rec_item in rec_list:
        rebuilt_src += rec_item["chunk_text"]

    assert rebuilt_src == src_text


def assert_bounded_export_rebuild(
    src_text: str,
    rec_list: list[dict],
) -> None:
    """Run all bounded export reconstruction checks together."""
    assert_export_offsets_match(src_text, rec_list)
    assert_bounded_recs_contiguous(src_text, rec_list)
    assert_bounded_recs_rebuild_source(src_text, rec_list)


def assert_bounded_recs_fit_token_limit(
    rec_list: list[dict],
    max_tokens: int,
) -> None:
    """Every exported bounded record should stay within max_tokens."""
    for rec_item in rec_list:
        tok_count = int(rec_item["token_count"])
        assert tok_count <= max_tokens
        assert tok_count == count_tokens(rec_item["chunk_text"])


def assert_text_in_recs(
    expected_text: str,
    rec_list: list[dict],
) -> None:
    """Check that important text appears somewhere in the exported records."""
    text_found = False

    for rec_item in rec_list:
        if expected_text in rec_item["chunk_text"]:
            text_found = True

    assert text_found


def assert_fixed_word_overlap(
    rec_list: list[dict],
    overlap: int,
) -> None:
    """Adjacent fixed chunks should share overlap words."""
    rec_pos = 0

    while rec_pos < len(rec_list) - 1:
        left_words = split_words(rec_list[rec_pos]["chunk_text"])
        right_words = split_words(rec_list[rec_pos + 1]["chunk_text"])

        assert left_words[-overlap:] == right_words[:overlap]

        rec_pos += 1


# -----------------------------------------------------------------------------
# Chunk command
# -----------------------------------------------------------------------------


def test_chunk_command_writes_paragraph_csv(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "outputs" / "paragraph.csv"

    src_path = copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )
    src_text = src_path.read_text(encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert out_path.exists()

    row_list = load_csv_rows(out_path)

    assert_csv_layout(row_list)
    assert len(row_list) == 3

    assert row_list[0]["chunk_id"] == (
        "car_engines_essay_txt_d1ed6ede_paragraph_000"
    )
    assert row_list[1]["chunk_id"] == (
        "car_engines_essay_txt_d1ed6ede_paragraph_001"
    )
    assert row_list[2]["chunk_id"] == (
        "car_engines_essay_txt_d1ed6ede_paragraph_002"
    )

    assert row_list[0]["chunk_index"] == "0"
    assert row_list[1]["chunk_index"] == "1"
    assert row_list[2]["chunk_index"] == "2"

    for row_item in row_list:
        assert row_item["source_file"] == "car_engines_essay.txt"
        assert row_item["source_path"] == "car_engines_essay.txt"
        assert row_item["method"] == "paragraph"
        assert row_item["chunk_size"] == ""
        assert row_item["overlap"] == ""
        assert row_item["token_count"] == ""
        assert row_item["max_tokens"] == ""
        assert row_item["tokenizer"] == ""

    assert_export_offsets_match(src_text, row_list)

    assert "Method: paragraph" in cli_res.output
    assert "Chunk size:" not in cli_res.output
    assert "Maximum tokens:" not in cli_res.output
    assert "Tokenizer:" not in cli_res.output
    assert "Tokenized chunks:" not in cli_res.output
    assert "Processed files: 1" in cli_res.output
    assert "Total chunks: 3" in cli_res.output


def test_chunk_command_merges_markdown_headings_for_paragraph(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.csv"
    doc_dir.mkdir()

    src_path = doc_dir / "notes.md"
    src_text = (
        "# Notes\n\n"
        "This paragraph should stay attached to the heading."
    )
    src_path.write_text(src_text, encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    row_list = load_csv_rows(out_path)

    assert_csv_layout(row_list)
    assert len(row_list) == 1
    assert row_list[0]["source_file"] == "notes.md"
    assert row_list[0]["method"] == "paragraph"
    assert row_list[0]["chunk_text"] == src_text
    assert row_list[0]["chunk_index"] == "0"
    assert_export_offsets_match(src_text, row_list)

    assert "Total chunks: 1" in cli_res.output


def test_chunk_command_does_not_merge_heading_text_for_txt(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.csv"
    doc_dir.mkdir()

    src_path = doc_dir / "notes.txt"
    src_text = (
        "# Notes\n\n"
        "This paragraph should stay separate in a txt file."
    )
    src_path.write_text(src_text, encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    row_list = load_csv_rows(out_path)

    assert_csv_layout(row_list)
    assert len(row_list) == 2
    assert row_list[0]["chunk_text"] == "# Notes"
    assert row_list[1]["chunk_text"] == (
        "This paragraph should stay separate in a txt file."
    )
    assert_export_offsets_match(src_text, row_list)

    assert "Total chunks: 2" in cli_res.output


def test_chunk_command_writes_fixed_jsonl(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "outputs" / "fixed.jsonl"

    src_path = copy_fxtr(
        doc_dir,
        "bird_watching_article.txt",
    )
    src_text = src_path.read_text(encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "fixed",
            "--chunk-size",
            "12",
            "--overlap",
            "3",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    rec_list = load_jsonl_recs(out_path)

    assert_jsonl_layout(rec_list)
    assert len(rec_list) == 10

    rec_pos = 0

    while rec_pos < len(rec_list):
        rec_item = rec_list[rec_pos]

        assert rec_item["chunk_index"] == rec_pos
        assert rec_item["method"] == "fixed"
        assert rec_item["chunk_size"] == 12
        assert rec_item["overlap"] == 3
        assert rec_item["token_count"] is None
        assert rec_item["max_tokens"] is None
        assert rec_item["tokenizer"] is None
        assert rec_item["word_count"] <= 12

        rec_pos += 1

    assert rec_list[0]["chunk_id"] == (
        "bird_watching_article_txt_f6b321c9_fixed_cs12_ov3_000"
    )

    assert_fixed_word_overlap(rec_list, overlap=3)
    assert_export_offsets_match(src_text, rec_list)

    assert "Method: fixed" in cli_res.output
    assert "Chunk size: 12" in cli_res.output
    assert "Overlap: 3" in cli_res.output
    assert "Maximum tokens:" not in cli_res.output
    assert "Tokenized chunks:" not in cli_res.output
    assert "Total chunks: 10" in cli_res.output


def test_chunk_command_writes_bounded_jsonl(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "outputs" / "bounded.jsonl"
    max_tokens = 20

    src_path = copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
        "car_engines_essay.txt",
    )
    src_text = src_path.read_text(encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            str(max_tokens),
            "--tokenizer",
            "cl100k_base",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert out_path.exists()

    rec_list = load_jsonl_recs(out_path)

    assert_jsonl_layout(rec_list)
    assert len(rec_list) >= 3

    for rec_item in rec_list:
        assert rec_item["method"] == "bounded-paragraph"
        assert rec_item["chunk_size"] is None
        assert rec_item["overlap"] is None
        assert rec_item["max_tokens"] == max_tokens
        assert rec_item["tokenizer"] == "cl100k_base"

    assert rec_list[0]["chunk_id"] == (
        "car_engines_essay_txt_d1ed6ede_"
        "bounded_paragraph_mt20_tk_cl100k_base_000"
    )

    assert_text_in_recs("Car Engines", rec_list)
    assert_text_in_recs("Diesel engines", rec_list)
    assert_text_in_recs("Fully electric cars", rec_list)
    assert_bounded_recs_fit_token_limit(rec_list, max_tokens)
    assert_bounded_export_rebuild(src_text, rec_list)

    assert "Method: bounded-paragraph" in cli_res.output
    assert "Maximum tokens: 20" in cli_res.output
    assert "Tokenizer: cl100k_base" in cli_res.output
    assert "Chunk size:" not in cli_res.output
    assert "Overlap:" not in cli_res.output
    assert "Tokenized chunks:" in cli_res.output
    assert "Average tokens per chunk:" in cli_res.output
    assert "Shortest token count:" in cli_res.output
    assert "Longest token count:" in cli_res.output


def test_chunk_command_writes_bounded_csv_that_rebuilds_source(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "bounded.csv"
    max_tokens = 100
    doc_dir.mkdir()

    src_path = doc_dir / "spacing_notes.txt"
    src_text = (
        "\n"
        "First paragraph starts after a blank line.\n\n"
        "Second paragraph has enough detail for the test.\n\n"
        "Final paragraph ends before one trailing line."
        "\n"
    )
    src_path.write_text(src_text, encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            str(max_tokens),
            "--tokenizer",
            "cl100k_base",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert out_path.exists()

    row_list = load_csv_rows(out_path)

    assert_csv_layout(row_list)
    assert len(row_list) >= 3

    for row_item in row_list:
        assert row_item["method"] == "bounded-paragraph"
        assert row_item["chunk_size"] == ""
        assert row_item["overlap"] == ""
        assert row_item["max_tokens"] == str(max_tokens)
        assert row_item["tokenizer"] == "cl100k_base"

    assert_text_in_recs("First paragraph", row_list)
    assert_text_in_recs("Second paragraph", row_list)
    assert_text_in_recs("Final paragraph", row_list)
    assert_bounded_recs_fit_token_limit(row_list, max_tokens)
    assert_bounded_export_rebuild(src_text, row_list)

    assert "Method: bounded-paragraph" in cli_res.output
    assert "Total chunks:" in cli_res.output


def test_chunk_command_preserves_crlf_in_bounded_export(
    tmp_path: Path,
) -> None:
    crlf_doc_dir = tmp_path / "docs"
    crlf_out_path = tmp_path / "bounded_crlf.csv"
    crlf_doc_dir.mkdir()

    crlf_src_path = crlf_doc_dir / "line_endings.txt"
    crlf_src_text = (
        "First paragraph keeps its line ending.\r\n\r\n"
        "Second paragraph also stays unchanged.\r\n"
    )
    crlf_src_path.write_bytes(crlf_src_text.encode("utf-8"))

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(crlf_doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            "100",
            "--tokenizer",
            "cl100k_base",
            "--out",
            str(crlf_out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    row_list = load_csv_rows(crlf_out_path)

    assert row_list
    assert_bounded_export_rebuild(crlf_src_text, row_list)
    assert "\r\n" in "".join(row_item["chunk_text"] for row_item in row_list)


def test_chunk_command_merges_markdown_headings_for_bounded_paragraph(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "bounded.jsonl"
    max_tokens = 50
    doc_dir.mkdir()

    src_path = doc_dir / "site_notes.md"
    src_text = (
        "## Site Walk\n\n"
        "The team checked the north entrance."
    )
    src_path.write_text(src_text, encoding="utf-8")

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            str(max_tokens),
            "--tokenizer",
            "cl100k_base",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    rec_list = load_jsonl_recs(out_path)

    assert_jsonl_layout(rec_list)
    assert len(rec_list) >= 1

    for rec_item in rec_list:
        assert rec_item["source_file"] == "site_notes.md"
        assert rec_item["method"] == "bounded-paragraph"
        assert rec_item["max_tokens"] == max_tokens
        assert rec_item["tokenizer"] == "cl100k_base"

    assert_text_in_recs("## Site Walk", rec_list)
    assert_text_in_recs("north entrance", rec_list)
    assert_bounded_recs_fit_token_limit(rec_list, max_tokens)
    assert_bounded_export_rebuild(src_text, rec_list)

    assert "Total chunks:" in cli_res.output
    assert "Tokenized chunks:" in cli_res.output


def test_chunk_command_creates_unique_ids_for_same_stem(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    out_path = tmp_path / "chunks.csv"

    (doc_dir / "notes.txt").write_text(
        "Typed notes from the morning lecture.",
        encoding="utf-8",
    )
    (doc_dir / "notes.md").write_text(
        "# Notes\n\nQuick meeting list from after school.",
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    row_list = load_csv_rows(out_path)
    chnk_ids = []

    for row_item in row_list:
        chnk_ids.append(row_item["chunk_id"])

    assert len(row_list) == 2
    assert len(chnk_ids) == len(set(chnk_ids))
    assert "notes_txt_83348d28_paragraph_000" in chnk_ids
    assert "notes_md_12dfac2a_paragraph_000" in chnk_ids

    md_rows = []

    for row_item in row_list:
        if row_item["source_file"] == "notes.md":
            md_rows.append(row_item)

    assert len(md_rows) == 1
    assert md_rows[0]["chunk_text"] == (
        "# Notes\n\nQuick meeting list from after school."
    )


def test_chunk_command_processes_nested_files_in_order(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    nest_dir = doc_dir / "class_notes"
    out_path = tmp_path / "chunks.csv"

    copy_fxtr(
        nest_dir,
        "data_structures_lecture_note.txt",
    )
    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    row_list = load_csv_rows(out_path)
    src_paths = []

    for row_item in row_list:
        src_paths.append(row_item["source_path"])

    assert src_paths[0] == "car_engines_essay.txt"
    assert src_paths[1] == "car_engines_essay.txt"
    assert src_paths[2] == "car_engines_essay.txt"
    assert src_paths[-1] == (
        "class_notes/data_structures_lecture_note.txt"
    )
    assert "Processed files: 2" in cli_res.output
    assert "Total chunks: 4" in cli_res.output


def test_chunk_command_reports_skipped_files(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.csv"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )
    copy_fxtr(doc_dir, "blank_document.txt")

    (doc_dir / "site_photo.png").write_text(
        "not a supported text file",
        encoding="utf-8",
    )
    (doc_dir / "corrupt_notes.txt").write_bytes(
        b"\xff\xfe\x00\x00"
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert "Processed files: 1" in cli_res.output
    assert "Unsupported files skipped: 1" in cli_res.output
    assert "Empty files skipped: 1" in cli_res.output
    assert "Read error files skipped: 1" in cli_res.output
    assert "Total chunks: 1" in cli_res.output


def test_chunk_command_creates_output_directories(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "one" / "two" / "chunks.csv"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert out_path.exists()


def test_chunk_command_accepts_normalized_method(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.jsonl"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "  FIXED  ",
            "--chunk-size",
            "50",
            "--overlap",
            "10",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    rec_list = load_jsonl_recs(out_path)

    assert len(rec_list) == 1
    assert rec_list[0]["method"] == "fixed"
    assert rec_list[0]["chunk_size"] == 50
    assert rec_list[0]["overlap"] == 10
    assert rec_list[0]["chunk_id"] == (
        "data_structures_lecture_note_txt_"
        "cbe2a65a_fixed_cs50_ov10_000"
    )


def test_chunk_command_normalizes_tokenizer_name(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.jsonl"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            "20",
            "--tokenizer",
            "  cl100k_base  ",
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output

    rec_list = load_jsonl_recs(out_path)

    assert rec_list
    assert rec_list[0]["tokenizer"] == "cl100k_base"
    assert rec_list[0]["chunk_id"].startswith(
        "data_structures_lecture_note_txt_"
        "cbe2a65a_bounded_paragraph_mt20_tk_cl100k_base_"
    )
    assert "Tokenizer: cl100k_base" in cli_res.output


def test_chunk_command_rejects_missing_input_directory(
    tmp_path: Path,
) -> None:
    miss_dir = tmp_path / "missing_docs"

    cli_res = cli_runner.invoke(
        app,
        ["chunk", str(miss_dir)],
    )

    assert cli_res.exit_code == 1
    assert "Error: Input path does not exist:" in cli_res.output


def test_chunk_command_rejects_unknown_method(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "sentence",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Method must be 'paragraph', "
        "'bounded-paragraph', or 'fixed'."
        in cli_res.output
    )


def test_chunk_command_rejects_invalid_fixed_overlap(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"

    copy_fxtr(
        doc_dir,
        "bird_watching_article.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "fixed",
            "--chunk-size",
            "12",
            "--overlap",
            "12",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Overlap must be smaller than chunk size."
        in cli_res.output
    )


def test_chunk_command_rejects_zero_max_tokens(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"

    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            "0",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Maximum tokens must be greater than 0."
        in cli_res.output
    )


def test_chunk_command_rejects_unknown_tokenizer(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"

    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--tokenizer",
            "not_a_real_tokenizer",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Unknown tokenizer encoding: "
        "not_a_real_tokenizer"
        in cli_res.output
    )


def test_chunk_command_rejects_empty_tokenizer(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"

    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--tokenizer",
            "   ",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Tokenizer name cannot be empty."
        in cli_res.output
    )


def test_chunk_command_rejects_unsupported_output(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.txt"

    copy_fxtr(
        doc_dir,
        "data_structures_lecture_note.txt",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--out",
            str(out_path),
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Unsupported output file type: .txt"
        in cli_res.output
    )
    assert not out_path.exists()


# -----------------------------------------------------------------------------
# Inspect command
# -----------------------------------------------------------------------------


def test_inspect_command_summarizes_csv(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.csv"

    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    chnk_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "paragraph",
            "--out",
            str(out_path),
        ],
    )

    assert chnk_res.exit_code == 0

    cli_res = cli_runner.invoke(
        app,
        [
            "inspect",
            str(out_path),
            "--limit",
            "2",
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert "Format: CSV" in cli_res.output
    assert "Total chunks: 3" in cli_res.output
    assert "Source files: 1" in cli_res.output
    assert "Methods: paragraph" in cli_res.output
    assert "Preview records: 2" in cli_res.output
    assert "car_engines_essay_txt_d1ed6ede_paragraph_000" in cli_res.output
    assert (
        "car_engines_essay_txt_d1ed6ede_paragraph_002"
        not in cli_res.output
    )

    assert "Tokenized chunks:" not in cli_res.output
    assert "Average tokens per chunk:" not in cli_res.output
    assert "Shortest token count:" not in cli_res.output
    assert "Longest token count:" not in cli_res.output


def test_inspect_counts_duplicate_names_by_source_path(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    nest_dir = doc_dir / "nested"
    out_path = tmp_path / "chunks.csv"

    nest_dir.mkdir(parents=True)

    (doc_dir / "alpha.txt").write_text(
        "Root alpha paragraph.",
        encoding="utf-8",
    )
    (nest_dir / "alpha.txt").write_text(
        "Nested alpha paragraph.",
        encoding="utf-8",
    )

    chnk_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--out",
            str(out_path),
        ],
    )

    assert chnk_res.exit_code == 0

    cli_res = cli_runner.invoke(
        app,
        [
            "inspect",
            str(out_path),
            "--limit",
            "0",
        ],
    )

    assert cli_res.exit_code == 0
    assert "Total chunks: 2" in cli_res.output
    assert "Source files: 2" in cli_res.output


def test_inspect_command_summarizes_bounded_jsonl(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "bounded.jsonl"

    copy_fxtr(
        doc_dir,
        "car_engines_essay.txt",
    )

    chnk_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "bounded-paragraph",
            "--max-tokens",
            "20",
            "--out",
            str(out_path),
        ],
    )

    assert chnk_res.exit_code == 0

    cli_res = cli_runner.invoke(
        app,
        [
            "inspect",
            str(out_path),
            "--limit",
            "1",
        ],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert "Format: JSONL" in cli_res.output
    assert "Methods: bounded-paragraph" in cli_res.output

    assert "Tokenized chunks:" in cli_res.output
    assert "Average tokens per chunk:" in cli_res.output
    assert "Shortest token count:" in cli_res.output
    assert "Longest token count:" in cli_res.output

    assert "Preview records: 1" in cli_res.output
    assert " words | " in cli_res.output
    assert " tokens | " in cli_res.output


def test_inspect_command_respects_zero_limit(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "chunks.jsonl"

    copy_fxtr(
        doc_dir,
        "bird_watching_article.txt",
    )

    chnk_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--method",
            "fixed",
            "--chunk-size",
            "12",
            "--overlap",
            "3",
            "--out",
            str(out_path),
        ],
    )

    assert chnk_res.exit_code == 0

    cli_res = cli_runner.invoke(
        app,
        [
            "inspect",
            str(out_path),
            "--limit",
            "0",
        ],
    )

    assert cli_res.exit_code == 0
    assert "Total chunks: 10" in cli_res.output
    assert "Preview records:" not in cli_res.output
    assert "Tokenized chunks:" not in cli_res.output
    assert "Average tokens per chunk:" not in cli_res.output


def test_inspect_command_rejects_negative_limit(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.jsonl"
    out_path.write_text(
        '{"chunk_id": "sample_000"}\n',
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        [
            "inspect",
            str(out_path),
            "--limit",
            "-1",
        ],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Preview limit must be greater than or equal to 0."
        in cli_res.output
    )


def test_inspect_command_handles_empty_csv(
    tmp_path: Path,
) -> None:
    doc_dir = tmp_path / "docs"
    out_path = tmp_path / "empty.csv"

    copy_fxtr(
        doc_dir,
        "blank_document.txt",
    )

    chnk_res = cli_runner.invoke(
        app,
        [
            "chunk",
            str(doc_dir),
            "--out",
            str(out_path),
        ],
    )

    assert chnk_res.exit_code == 0

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 0
    assert "Total chunks: 0" in cli_res.output
    assert "Source files: 0" in cli_res.output
    assert "Methods: <none>" in cli_res.output
    assert "Average words per chunk: 0.0" in cli_res.output
    assert "Shortest chunk: 0 words" in cli_res.output
    assert "Longest chunk: 0 words" in cli_res.output
    assert "Preview records:" not in cli_res.output
    assert "Tokenized chunks:" not in cli_res.output


def test_inspect_command_uses_singular_count_labels(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "single_chunk.jsonl"
    rec_item = {}

    for field in EXPORT_FIELDS:
        rec_item[field] = None

    rec_item.update(
        {
            "chunk_id": "single_000",
            "source_file": "single.txt",
            "source_path": "single.txt",
            "method": "bounded-paragraph",
            "token_count": 1,
            "max_tokens": 10,
            "tokenizer": "cl100k_base",
            "chunk_index": 0,
            "chunk_text": "Hello",
            "start_char": 0,
            "end_char": 5,
            "char_count": 5,
            "word_count": 1,
        }
    )
    out_path.write_text(
        json.dumps(rec_item) + "\n",
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 0
    assert "Shortest chunk: 1 word" in cli_res.output
    assert "Longest chunk: 1 word" in cli_res.output
    assert "Shortest token count: 1 token" in cli_res.output
    assert "Longest token count: 1 token" in cli_res.output


def test_inspect_command_skips_blank_jsonl_lines(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.jsonl"
    first_rec = {
        "chunk_id": "first_000",
        "source_file": "first.txt",
        "source_path": "first.txt",
        "method": "paragraph",
        "chunk_index": 0,
        "chunk_text": "Alpha",
        "word_count": 1,
    }
    second_rec = {
        "chunk_id": "second_000",
        "source_file": "second.txt",
        "source_path": "second.txt",
        "method": "paragraph",
        "chunk_index": 0,
        "chunk_text": "Beta",
        "word_count": 1,
    }
    out_path.write_text(
        json.dumps(first_rec)
        + "\n\n   \n"
        + json.dumps(second_rec)
        + "\n",
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path), "--limit", "0"],
    )

    assert cli_res.exit_code == 0, cli_res.output
    assert "Format: JSONL" in cli_res.output
    assert "Total chunks: 2" in cli_res.output
    assert "Source files: 2" in cli_res.output


def test_inspect_command_rejects_missing_file(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "missing.csv"

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 1
    assert "Error: Output file does not exist:" in cli_res.output


def test_inspect_command_rejects_directory_path(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.csv"
    out_path.mkdir()

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 1
    assert "Error: Output path is not a file:" in cli_res.output


def test_inspect_command_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.txt"
    out_path.write_text(
        "not an exported chunk file",
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 1
    assert (
        "Error: Unsupported output file type: .txt"
        in cli_res.output
    )


def test_inspect_command_rejects_invalid_jsonl(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.jsonl"
    out_path.write_text(
        '{"chunk_id": "ok"}\ninvalid json\n',
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 1
    assert "Error: Invalid JSONL at line 2" in cli_res.output


def test_inspect_rejects_non_object_jsonl_record(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "chunks.jsonl"
    out_path.write_text(
        '{"chunk_id": "ok"}\n["not", "an", "object"]\n',
        encoding="utf-8",
    )

    cli_res = cli_runner.invoke(
        app,
        ["inspect", str(out_path)],
    )

    assert cli_res.exit_code == 1
    assert (
        "Invalid JSONL at line 2: expected object"
        in cli_res.output
    )