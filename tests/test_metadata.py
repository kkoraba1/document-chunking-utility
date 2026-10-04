"""Tests for chunk metadata construction."""

from pathlib import Path

import pytest

from document_chunking_utility.chunkers import (
    TextChunk,
    chunk_by_bounded_paragraph,
    chunk_by_fixed_words,
    chunk_by_paragraph,
)
from document_chunking_utility.io import SourceDocument
from document_chunking_utility.metadata import (
    ChunkRecord,
    build_chunk_id,
    build_chunk_record,
    build_chunk_records,
    chunk_record_to_dict,
    chunk_records_to_dicts,
)
from document_chunking_utility.tokenizer import count_tokens

FIXTR_DIR = Path(__file__).parent / "fixtures" / "chunker_inputs"

REQ_FIELDS = {
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
}


def read_sample(file_name: str) -> str:
    """Read a fixture document without changing its line endings."""
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


def assert_record_counts_are_correct(rec_item: ChunkRecord) -> None:
    """Check that character and word counts match the chunk text."""
    assert rec_item.char_count == len(rec_item.chunk_text)
    assert rec_item.word_count == len(rec_item.chunk_text.split())


def assert_record_offsets_match_source(
    src_doc: SourceDocument,
    rec_list: list[ChunkRecord],
) -> None:
    """Check that every record points to its exact source span."""
    for rec_item in rec_list:
        assert rec_item.start_char < rec_item.end_char

        src_slice = src_doc.text[
            rec_item.start_char:rec_item.end_char
        ]

        assert src_slice == rec_item.chunk_text
        assert_record_counts_are_correct(rec_item)


def assert_bounded_records_are_contiguous(
    src_doc: SourceDocument,
    rec_list: list[ChunkRecord],
) -> None:
    """Check that bounded records cover the source in order."""
    if not rec_list:
        assert src_doc.text.strip() == ""
        return

    next_start = 0

    for rec_item in rec_list:
        assert rec_item.start_char == next_start
        next_start = rec_item.end_char

    assert next_start == len(src_doc.text)


def assert_bounded_records_rebuild_source(
    src_doc: SourceDocument,
    rec_list: list[ChunkRecord],
) -> None:
    """Check that bounded records rebuild the full source."""
    rebuilt_text = ""

    for rec_item in rec_list:
        rebuilt_text += rec_item.chunk_text

    assert rebuilt_text == src_doc.text


def assert_bounded_record_reconstruction(
    src_doc: SourceDocument,
    rec_list: list[ChunkRecord],
) -> None:
    """Run all bounded record reconstruction checks."""
    assert_record_offsets_match_source(src_doc, rec_list)
    assert_bounded_records_are_contiguous(src_doc, rec_list)
    assert_bounded_records_rebuild_source(src_doc, rec_list)


def assert_bounded_records_fit_token_limit(
    rec_list: list[ChunkRecord],
    max_tok: int,
) -> None:
    """Check that bounded records stay inside the token limit."""
    for rec_item in rec_list:
        assert rec_item.token_count is not None
        assert rec_item.token_count <= max_tok


def assert_text_appears_in_records(
    expected_text: str,
    rec_list: list[ChunkRecord],
) -> None:
    """Check that important text appears somewhere in the records."""
    text_found = False

    for rec_item in rec_list:
        if expected_text in rec_item.chunk_text:
            text_found = True

    assert text_found


def assert_text_appears_in_dicts(
    expected_text: str,
    rec_list: list[dict],
) -> None:
    """Check that important text appears in dictionary records."""
    text_found = False

    for rec_item in rec_list:
        if expected_text in rec_item["chunk_text"]:
            text_found = True

    assert text_found


# -----------------------------------------------------------------------------
# Chunk ID construction
# -----------------------------------------------------------------------------


def test_build_chunk_id_uses_zero_padded_chunk_index() -> None:
    assert build_chunk_id(
        source_path="semester/week_04/lecture_notes.txt",
        chunk_index=0,
        method="paragraph",
    ) == "semester_week_04_lecture_notes_txt_1bd2f563_paragraph_000"

    assert build_chunk_id(
        source_path="semester/week_04/lecture_notes.txt",
        chunk_index=17,
        method="paragraph",
    ) == "semester_week_04_lecture_notes_txt_1bd2f563_paragraph_017"


def test_build_chunk_id_normalizes_spaces_punctuation_and_case() -> None:
    assert build_chunk_id(
        source_path="Drafts/Office Hours - FINAL Copy.md",
        chunk_index=3,
        method="paragraph",
    ) == "drafts_office_hours_final_copy_md_20bb7e91_paragraph_003"


def test_build_chunk_id_handles_nested_windows_style_paths() -> None:
    assert build_chunk_id(
        source_path=r"Campus\Spring Notes\Professor Schedule.TXT",
        chunk_index=7,
        method="paragraph",
    ) == "campus_spring_notes_professor_schedule_txt_4b0a7643_paragraph_007"


def test_build_chunk_id_falls_back_when_filename_has_no_letters() -> None:
    assert build_chunk_id(
        source_path="!!!.txt",
        chunk_index=2,
        method="paragraph",
    ) == "document_txt_818ae954_paragraph_002"


def test_build_chunk_id_distinguishes_files_with_different_extensions() -> None:
    txt_id = build_chunk_id(
        "notes.txt",
        0,
        method="paragraph",
    )
    md_id = build_chunk_id(
        "notes.md",
        0,
        method="paragraph",
    )

    assert txt_id == "notes_txt_83348d28_paragraph_000"
    assert md_id == "notes_md_12dfac2a_paragraph_000"
    assert txt_id != md_id


def test_build_chunk_id_distinguishes_paths_with_similar_slugs() -> None:
    hyphen_id = build_chunk_id(
        "drafts/a-b.txt",
        0,
        method="paragraph",
    )
    under_id = build_chunk_id(
        "drafts/a_b.txt",
        0,
        method="paragraph",
    )

    assert hyphen_id == "drafts_a_b_txt_fd4cdf0f_paragraph_000"
    assert under_id == "drafts_a_b_txt_0f6ca887_paragraph_000"
    assert hyphen_id != under_id


# -----------------------------------------------------------------------------
# Single record construction
# -----------------------------------------------------------------------------


def test_build_chunk_record_returns_chunk_record_object() -> None:
    src_doc = make_document("data_structures_lecture_note.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="paragraph",
    )

    assert isinstance(rec_item, ChunkRecord)


def test_build_chunk_record_preserves_source_document_fields() -> None:
    src_doc = make_document(
        "construction_meeting_notes.md",
        source_path=(
            "campus/building_project/"
            "construction_meeting_notes.md"
        ),
    )
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="paragraph",
    )

    assert rec_item.source_file == "construction_meeting_notes.md"
    assert rec_item.source_path == (
        "campus/building_project/construction_meeting_notes.md"
    )
    assert rec_item.chunk_id == (
        "campus_building_project_construction_meeting_notes_"
        "md_3e372ed3_paragraph_000"
    )


def test_build_chunk_record_preserves_chunk_position_and_text() -> None:
    src_doc = make_document("car_engines_essay.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[1]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="paragraph",
    )

    assert rec_item.method == "paragraph"
    assert rec_item.chunk_index == 1
    assert rec_item.chunk_text == chnk_item.chunk_text
    assert rec_item.start_char == chnk_item.start_char
    assert rec_item.end_char == chnk_item.end_char
    assert "Diesel engines" in rec_item.chunk_text
    assert_record_counts_are_correct(rec_item)


def test_paragraph_record_has_empty_chunk_settings() -> None:
    src_doc = make_document("car_engines_essay.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="paragraph",
    )

    assert rec_item.chunk_size is None
    assert rec_item.overlap is None
    assert rec_item.token_count is None
    assert rec_item.max_tokens is None
    assert rec_item.tokenizer is None


def test_fixed_record_stores_fixed_chunk_settings() -> None:
    src_doc = make_document("bird_watching_article.txt")
    chnk_item = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=12,
        overlap=3,
    )[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="fixed",
        chunk_size=12,
        overlap=3,
    )

    assert rec_item.method == "fixed"
    assert rec_item.chunk_size == 12
    assert rec_item.overlap == 3
    assert rec_item.token_count is None
    assert rec_item.max_tokens is None
    assert rec_item.tokenizer is None
    assert_record_counts_are_correct(rec_item)


def test_bounded_record_stores_token_metadata() -> None:
    src_doc = make_document("data_structures_lecture_note.txt")
    chnk_item = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=20,
    )[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="bounded-paragraph",
        max_tokens=20,
        tokenizer="cl100k_base",
    )

    assert rec_item.method == "bounded-paragraph"
    assert rec_item.chunk_size is None
    assert rec_item.overlap is None
    assert rec_item.token_count == count_tokens(rec_item.chunk_text)
    assert rec_item.token_count <= 20
    assert rec_item.max_tokens == 20
    assert rec_item.tokenizer == "cl100k_base"
    assert_record_counts_are_correct(rec_item)


def test_bounded_record_requires_tokenizer() -> None:
    src_doc = make_document("data_structures_lecture_note.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    with pytest.raises(
        ValueError,
        match="Tokenizer is required",
    ):
        build_chunk_record(
            document=src_doc,
            chunk=chnk_item,
            method="bounded-paragraph",
            max_tokens=20,
        )


def test_bounded_record_requires_maximum_tokens() -> None:
    src_doc = make_document("data_structures_lecture_note.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    with pytest.raises(
        ValueError,
        match="Maximum tokens are required",
    ):
        build_chunk_record(
            document=src_doc,
            chunk=chnk_item,
            method="bounded-paragraph",
            tokenizer="cl100k_base",
        )


def test_build_chunk_record_handles_unicode_text() -> None:
    src_doc = make_document("unicode_japanese_notes.txt")
    chnk_item = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=7,
        overlap=2,
    )[1]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="fixed",
        chunk_size=7,
        overlap=2,
    )

    assert "東京" in rec_item.chunk_text

    assert (
        src_doc.text[rec_item.start_char:rec_item.end_char]
        == rec_item.chunk_text
    )

    assert_record_counts_are_correct(rec_item)


# -----------------------------------------------------------------------------
# Multiple record construction
# -----------------------------------------------------------------------------


def test_build_chunk_records_preserves_paragraph_order() -> None:
    src_doc = make_document("car_engines_essay.txt")
    chnk_list = chunk_by_paragraph(src_doc.text)

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )

    assert len(rec_list) == 3
    assert rec_list[0].chunk_index == 0
    assert rec_list[1].chunk_index == 1
    assert rec_list[2].chunk_index == 2
    assert rec_list[0].chunk_text.startswith("Car Engines")
    assert rec_list[1].chunk_text.startswith("Diesel engines")
    assert rec_list[2].chunk_text.startswith("Hybrid cars")
    assert_record_offsets_match_source(src_doc, rec_list)


def test_build_chunk_records_preserves_fixed_settings() -> None:
    src_doc = make_document("bird_watching_article.txt")
    chnk_list = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=12,
        overlap=3,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="fixed",
        chunk_size=12,
        overlap=3,
    )

    assert len(rec_list) == 10

    for rec_item in rec_list:
        assert rec_item.chunk_size == 12
        assert rec_item.overlap == 3
        assert rec_item.method == "fixed"

    assert_record_offsets_match_source(src_doc, rec_list)


def test_build_chunk_records_preserves_bounded_settings() -> None:
    src_doc = make_document("car_engines_essay.txt")
    max_tok = 20

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) > 3

    for rec_item in rec_list:
        assert rec_item.method == "bounded-paragraph"
        assert rec_item.chunk_size is None
        assert rec_item.overlap is None
        assert rec_item.token_count is not None
        assert rec_item.token_count <= max_tok
        assert rec_item.max_tokens == max_tok
        assert rec_item.tokenizer == "cl100k_base"

    assert_bounded_record_reconstruction(src_doc, rec_list)


def test_build_chunk_records_creates_deterministic_unique_ids() -> None:
    src_doc = make_document(
        "construction_meeting_notes.md",
        source_path="project_notes/construction_meeting_notes.md",
    )
    chnk_list = chunk_by_paragraph(src_doc.text)

    first_run = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )
    second_run = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )

    first_ids = []

    for rec_item in first_run:
        first_ids.append(rec_item.chunk_id)

    second_ids = []

    for rec_item in second_run:
        second_ids.append(rec_item.chunk_id)

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [
        (
            "project_notes_construction_meeting_notes_md_"
            "50fa3337_paragraph_000"
        ),
        (
            "project_notes_construction_meeting_notes_md_"
            "50fa3337_paragraph_001"
        ),
        (
            "project_notes_construction_meeting_notes_md_"
            "50fa3337_paragraph_002"
        ),
        (
            "project_notes_construction_meeting_notes_md_"
            "50fa3337_paragraph_003"
        ),
        (
            "project_notes_construction_meeting_notes_md_"
            "50fa3337_paragraph_004"
        ),
    ]


def test_build_chunk_records_uses_source_path_to_avoid_collisions() -> None:
    src_text = read_sample("data_structures_lecture_note.txt")

    chnk_list = [
        TextChunk(
            chunk_index=0,
            chunk_text=src_text,
            start_char=0,
            end_char=len(src_text),
        )
    ]

    first_doc = SourceDocument(
        source_file="lecture_note.txt",
        source_path="section_a/lecture_note.txt",
        text=src_text,
    )
    second_doc = SourceDocument(
        source_file="lecture_note.txt",
        source_path="section_b/lecture_note.txt",
        text=src_text,
    )

    first_records = build_chunk_records(
        document=first_doc,
        chunks=chnk_list,
        method="paragraph",
    )
    second_records = build_chunk_records(
        document=second_doc,
        chunks=chnk_list,
        method="paragraph",
    )

    assert first_records[0].chunk_id == (
        "section_a_lecture_note_txt_14b58c5f_paragraph_000"
    )
    assert second_records[0].chunk_id == (
        "section_b_lecture_note_txt_9a731654_paragraph_000"
    )
    assert first_records[0].chunk_id != second_records[0].chunk_id


def test_build_chunk_records_returns_empty_list() -> None:
    src_doc = make_document("blank_document.txt")

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=[],
        method="paragraph",
    )

    assert rec_list == []


# -----------------------------------------------------------------------------
# Dictionary conversion
# -----------------------------------------------------------------------------


def test_chunk_record_to_dict_contains_required_schema_fields() -> None:
    src_doc = make_document("data_structures_lecture_note.txt")
    chnk_item = chunk_by_paragraph(src_doc.text)[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="paragraph",
    )

    rec_dict = chunk_record_to_dict(rec_item)

    assert set(rec_dict.keys()) == REQ_FIELDS


def test_chunk_record_to_dict_preserves_metadata_values() -> None:
    src_doc = make_document("bird_watching_article.txt")
    chnk_item = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=12,
        overlap=3,
    )[0]

    rec_item = build_chunk_record(
        document=src_doc,
        chunk=chnk_item,
        method="fixed",
        chunk_size=12,
        overlap=3,
    )

    rec_dict = chunk_record_to_dict(rec_item)

    assert rec_dict["chunk_id"] == (
        "bird_watching_article_txt_f6b321c9_fixed_cs12_ov3_000"
    )
    assert rec_dict["source_file"] == "bird_watching_article.txt"
    assert rec_dict["source_path"] == "bird_watching_article.txt"
    assert rec_dict["method"] == "fixed"
    assert rec_dict["chunk_size"] == 12
    assert rec_dict["overlap"] == 3
    assert rec_dict["token_count"] is None
    assert rec_dict["max_tokens"] is None
    assert rec_dict["tokenizer"] is None
    assert rec_dict["chunk_text"] == rec_item.chunk_text
    assert rec_dict["char_count"] == len(rec_item.chunk_text)
    assert rec_dict["word_count"] == len(rec_item.chunk_text.split())


def test_chunk_record_to_dict_preserves_bounded_offsets_and_text() -> None:
    src_text = (
        "\n\n"
        "# Weekly Notes\n\n"
        "The first section has useful information for the project.\n\n"
        "The second section has follow-up details."
        "\n"
    )

    src_doc = SourceDocument(
        source_file="weekly_notes.md",
        source_path="notes/weekly_notes.md",
        text=src_text,
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=100,
        merge_markdown_headings=True,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=100,
        tokenizer="cl100k_base",
    )

    rec_dict = chunk_record_to_dict(rec_list[0])

    assert rec_dict["start_char"] == rec_list[0].start_char
    assert rec_dict["end_char"] == rec_list[0].end_char
    assert rec_dict["chunk_text"] == rec_list[0].chunk_text

    src_slice = src_doc.text[
        rec_dict["start_char"]:rec_dict["end_char"]
    ]

    assert src_slice == rec_dict["chunk_text"]


def test_chunk_records_to_dicts_preserves_order() -> None:
    src_doc = make_document("car_engines_essay.txt")
    chnk_list = chunk_by_paragraph(src_doc.text)

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )

    dict_list = chunk_records_to_dicts(rec_list)

    chnk_ids = []
    chnk_indexes = []

    for rec_item in dict_list:
        chnk_ids.append(rec_item["chunk_id"])
        chnk_indexes.append(rec_item["chunk_index"])

    assert chnk_ids == [
        "car_engines_essay_txt_d1ed6ede_paragraph_000",
        "car_engines_essay_txt_d1ed6ede_paragraph_001",
        "car_engines_essay_txt_d1ed6ede_paragraph_002",
    ]
    assert chnk_indexes == [0, 1, 2]


def test_bounded_chunk_records_to_dicts_rebuild_source() -> None:
    src_text = (
        "\n"
        "First paragraph starts after one blank line.\n\n"
        "Second paragraph has enough words to be useful.\n\n"
        "Final paragraph ends before a trailing blank line."
        "\n"
    )

    src_doc = SourceDocument(
        source_file="spacing_notes.txt",
        source_path="spacing_notes.txt",
        text=src_text,
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=100,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=100,
        tokenizer="cl100k_base",
    )

    dict_list = chunk_records_to_dicts(rec_list)
    rebuilt_text = ""

    for rec_item in dict_list:
        src_slice = src_doc.text[
            rec_item["start_char"]:rec_item["end_char"]
        ]

        assert src_slice == rec_item["chunk_text"]

        rebuilt_text += rec_item["chunk_text"]

    assert_text_appears_in_dicts("First paragraph", dict_list)
    assert_text_appears_in_dicts("Second paragraph", dict_list)
    assert_text_appears_in_dicts("Final paragraph", dict_list)
    assert rebuilt_text == src_doc.text


def test_chunk_records_to_dicts_returns_empty_list() -> None:
    assert chunk_records_to_dicts([]) == []


# -----------------------------------------------------------------------------
# End-to-end metadata traceability
# -----------------------------------------------------------------------------


def test_paragraph_offsets_reconstruct_chunks() -> None:
    src_doc = make_document("professor_office_hours_windows.txt")
    chnk_list = chunk_by_paragraph(src_doc.text)

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="paragraph",
    )

    assert len(rec_list) == 4
    assert_record_offsets_match_source(src_doc, rec_list)

    for rec_item in rec_list:
        assert "\r\n" in rec_item.chunk_text


def test_fixed_offsets_reconstruct_chunks() -> None:
    src_doc = make_document("course_schedule_scramble.txt")

    chnk_list = chunk_by_fixed_words(
        src_doc.text,
        chunk_size=8,
        overlap=2,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="fixed",
        chunk_size=8,
        overlap=2,
    )

    assert len(rec_list) == 9
    assert rec_list[0].chunk_text.startswith("fall schedule draft")
    assert "math 241" in rec_list[3].chunk_text
    assert_record_offsets_match_source(src_doc, rec_list)


def test_bounded_offsets_reconstruct_chunks() -> None:
    src_doc = make_document("car_engines_essay.txt")
    max_tok = 20

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) > 3
    assert_text_appears_in_records("Car Engines", rec_list)
    assert_text_appears_in_records("Diesel engines", rec_list)
    assert_text_appears_in_records("Fully electric cars", rec_list)
    assert_bounded_record_reconstruction(src_doc, rec_list)
    assert_bounded_records_fit_token_limit(rec_list, max_tok)


def test_bounded_offsets_rebuild_source_with_heading_merge() -> None:
    src_text = (
        "\n\n"
        "# Project Notes\n\n"
        "## Site Walk\n\n"
        "The team reviewed the entrance and listed repair tasks.\n\n"
        "## Follow Up\n\n"
        "Students wrote the next steps after the meeting."
        "\n"
    )
    max_tok = 100

    src_doc = SourceDocument(
        source_file="project_notes.md",
        source_path="project_notes.md",
        text=src_text,
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
        merge_markdown_headings=True,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) >= 2
    assert_text_appears_in_records("# Project Notes", rec_list)
    assert_text_appears_in_records("## Follow Up", rec_list)
    assert_bounded_record_reconstruction(src_doc, rec_list)
    assert_bounded_records_fit_token_limit(rec_list, max_tok)


def test_bounded_offsets_rebuild_source_after_sentence_splitting() -> None:
    first_sent = (
        "The library opens early for students who need quiet study time."
    )
    second_sent = (
        "The computer lab stays open later during final exam week."
    )
    src_text = f"\n{first_sent} {second_sent}\n"

    src_doc = SourceDocument(
        source_file="student_services.txt",
        source_path="student_services.txt",
        text=src_text,
    )

    max_tok = max(
        count_tokens(first_sent),
        count_tokens(second_sent),
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) >= 2
    assert_text_appears_in_records(first_sent, rec_list)
    assert_text_appears_in_records(second_sent, rec_list)
    assert_bounded_record_reconstruction(src_doc, rec_list)
    assert_bounded_records_fit_token_limit(rec_list, max_tok)


def test_bounded_offsets_rebuild_source_after_word_fallback() -> None:
    src_text = (
        "\n"
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu"
        "\n"
    )
    max_tok = 3

    src_doc = SourceDocument(
        source_file="word_fallback.txt",
        source_path="word_fallback.txt",
        text=src_text,
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) > 1
    assert_text_appears_in_records("alpha", rec_list)
    assert_text_appears_in_records("mu", rec_list)
    assert_bounded_record_reconstruction(src_doc, rec_list)
    assert_bounded_records_fit_token_limit(rec_list, max_tok)


def test_bounded_offsets_rebuild_source_after_character_fallback() -> None:
    src_text = "\nsupercalifragilisticexpialidocious\n"
    max_tok = 3

    src_doc = SourceDocument(
        source_file="long_word.txt",
        source_path="long_word.txt",
        text=src_text,
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_doc.text,
        max_tokens=max_tok,
    )

    rec_list = build_chunk_records(
        document=src_doc,
        chunks=chnk_list,
        method="bounded-paragraph",
        max_tokens=max_tok,
        tokenizer="cl100k_base",
    )

    assert len(rec_list) > 1
    assert_bounded_record_reconstruction(src_doc, rec_list)
    assert_bounded_records_fit_token_limit(rec_list, max_tok)