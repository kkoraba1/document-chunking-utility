import json
from pathlib import Path

import pytest

from document_chunking_utility.chunkers import (
    chunk_by_bounded_paragraph,
    chunk_by_fixed_words,
    chunk_by_paragraph,
    chunk_text,
)
from document_chunking_utility.tokenizer import count_tokens
from document_chunking_utility.validation import ValidationError

FIXTR_DIR = Path(__file__).parent / "fixtures" / "chunker_inputs"
EXPECT_DIR = Path(__file__).parent / "fixtures" / "expected"


def read_sample(file_name: str) -> str:
    """Read a fixture document without changing its line endings."""
    file_path = FIXTR_DIR / file_name

    with open(file_path, "r", encoding="utf-8", newline="") as sample_file:
        return sample_file.read()


def read_expected(file_name: str) -> list[str]:
    """Read the expected chunk text for a fixture document."""
    file_path = EXPECT_DIR / file_name
    file_text = file_path.read_text(encoding="utf-8")

    return json.loads(file_text)


def chunk_texts(chnk_list) -> list[str]:
    """Return only the text from each chunk."""
    text_list = []

    for chnk_item in chnk_list:
        text_list.append(chnk_item.chunk_text)

    return text_list


def chunk_indexes(chnk_list) -> list[int]:
    """Return only the index from each chunk."""
    index_list = []

    for chnk_item in chnk_list:
        index_list.append(chnk_item.chunk_index)

    return index_list


def split_words(src_text: str) -> list[str]:
    """Split text the same simple way used for word counts."""
    return src_text.split()


def assert_indexes_are_sequential(chnk_list) -> None:
    """Check that chunk indexes begin at zero and stay in source order."""
    expected_indexes = list(range(len(chnk_list)))

    assert chunk_indexes(chnk_list) == expected_indexes


def assert_offsets_match_source(src_text: str, chnk_list) -> None:
    """Check that every chunk points to its exact source text span."""
    for chnk_item in chnk_list:
        assert chnk_item.start_char < chnk_item.end_char
        assert (
            src_text[chnk_item.start_char:chnk_item.end_char]
            == chnk_item.chunk_text
        )


def assert_bounded_chunks_are_contiguous(src_text: str, chnk_list) -> None:
    """Check that bounded chunks cover the source in order."""
    if not chnk_list:
        assert src_text.strip() == ""
        return

    next_start = 0

    for chnk_item in chnk_list:
        assert chnk_item.start_char == next_start
        next_start = chnk_item.end_char

    assert next_start == len(src_text)


def assert_bounded_chunks_rebuild_source(src_text: str, chnk_list) -> None:
    """Check that bounded chunks rebuild the full original text."""
    rebuilt_text = ""

    for chnk_item in chnk_list:
        rebuilt_text += chnk_item.chunk_text

    assert rebuilt_text == src_text


def assert_bounded_reconstruction(src_text: str, chnk_list) -> None:
    """Run all bounded paragraph reconstruction checks."""
    assert_offsets_match_source(src_text, chnk_list)
    assert_bounded_chunks_are_contiguous(src_text, chnk_list)
    assert_bounded_chunks_rebuild_source(src_text, chnk_list)


def assert_chunks_fit_token_limit(chnk_list, max_tok: int) -> None:
    """Check that every bounded chunk stays within the token limit."""
    for chnk_item in chnk_list:
        tok_count = count_tokens(chnk_item.chunk_text)

        assert tok_count <= max_tok


def assert_fixed_overlap(chnk_list, overlap: int) -> None:
    """Check that adjacent fixed chunks share the requested words."""
    chnk_pos = 0

    while chnk_pos < len(chnk_list) - 1:
        left_words = split_words(chnk_list[chnk_pos].chunk_text)
        right_words = split_words(chnk_list[chnk_pos + 1].chunk_text)

        assert left_words[-overlap:] == right_words[:overlap]

        chnk_pos += 1


def assert_text_appears_in_chunks(expected_text: str, chnk_list) -> None:
    """Check that important text appears somewhere in the chunks."""
    text_found = False

    for chnk_item in chnk_list:
        if expected_text in chnk_item.chunk_text:
            text_found = True

    assert text_found


# -----------------------------------------------------------------------------
# Paragraph chunking
# -----------------------------------------------------------------------------


def test_paragraph_chunking_splits_multi_section_document() -> None:
    src_text = read_sample("car_engines_essay.txt")
    expected_chunks = read_expected("car_engines_paragraph.json")

    chnk_list = chunk_by_paragraph(src_text)

    assert chunk_texts(chnk_list) == expected_chunks
    assert len(chnk_list) == 3
    assert chnk_list[0].chunk_text.startswith("Car Engines")
    assert "Diesel engines" in chnk_list[1].chunk_text
    assert "Fully electric cars" in chnk_list[2].chunk_text
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_treats_markdown_document_as_plain_text_by_default() -> None:
    src_text = read_sample("construction_meeting_notes.md")
    expected_chunks = read_expected(
        "construction_meeting_markdown_paragraph.json"
    )

    chnk_list = chunk_by_paragraph(src_text)

    assert chunk_texts(chnk_list) == expected_chunks
    assert len(chnk_list) == 5
    assert chnk_list[0].chunk_text == "# Construction Meeting Notes"
    assert chnk_list[2].chunk_text.startswith("## Site Walk")
    assert "- Rosa:" in chnk_list[-1].chunk_text
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_can_merge_markdown_heading_with_following_content() -> None:
    src_text = "# Notes\n\nThis paragraph should stay with the heading."

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 1
    assert chnk_list[0].chunk_text == src_text
    assert chnk_list[0].start_char == 0
    assert chnk_list[0].end_char == len(src_text)
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_can_merge_consecutive_markdown_headings() -> None:
    src_text = (
        "# Project Notes\n\n"
        "## Site Walk\n\n"
        "The team checked the north entrance and listed repair tasks."
    )

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 1
    assert chnk_list[0].chunk_text == src_text
    assert "# Project Notes" in chnk_list[0].chunk_text
    assert "## Site Walk" in chnk_list[0].chunk_text
    assert "north entrance" in chnk_list[0].chunk_text
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_keeps_final_heading_without_following_content() -> None:
    src_text = (
        "The opening paragraph has normal content.\n\n"
        "## Empty Section"
    )

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 2
    assert (
        chnk_list[0].chunk_text
        == "The opening paragraph has normal content."
    )
    assert chnk_list[1].chunk_text == "## Empty Section"
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_does_not_merge_hash_tag_text() -> None:
    src_text = "#not-a-heading\n\nThis paragraph should stay separate."

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 2
    assert chnk_list[0].chunk_text == "#not-a-heading"
    assert (
        chnk_list[1].chunk_text
        == "This paragraph should stay separate."
    )
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_merges_only_short_markdown_headings() -> None:
    long_heading = (
        "## This heading is intentionally written with many words so it "
        "does not get treated as a short section label"
    )
    src_text = f"{long_heading}\n\nThe paragraph should not merge."

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 2
    assert chnk_list[0].chunk_text == long_heading
    assert chnk_list[1].chunk_text == "The paragraph should not merge."
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_can_merge_heading_in_markdown_fixture() -> None:
    src_text = read_sample("construction_meeting_notes.md")

    chnk_list = chunk_by_paragraph(
        text=src_text,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 4
    assert chnk_list[0].chunk_text.startswith(
        "# Construction Meeting Notes"
    )
    assert "Date: Tuesday afternoon" in chnk_list[0].chunk_text
    assert chnk_list[1].chunk_text.startswith("## Site Walk")
    assert "- Rosa:" in chnk_list[-1].chunk_text
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_preserves_crlf_line_endings_and_offsets() -> None:
    src_text = read_sample("professor_office_hours_windows.txt")
    expected_chunks = read_expected(
        "professor_office_hours_windows_paragraph.json"
    )

    chnk_list = chunk_by_paragraph(src_text)

    assert "\r\n" in src_text
    assert chunk_texts(chnk_list) == expected_chunks
    assert len(chnk_list) == 4

    for chnk_item in chnk_list:
        assert "\r\n" in chnk_item.chunk_text

    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_paragraph_chunking_returns_empty_list_for_whitespace_only_file() -> None:
    src_text = read_sample("blank_document.txt")

    chnk_list = chunk_by_paragraph(src_text)

    assert chnk_list == []


def test_paragraph_chunking_is_deterministic_for_same_input() -> None:
    src_text = read_sample("car_engines_essay.txt")

    first_run = chunk_by_paragraph(src_text)
    second_run = chunk_by_paragraph(src_text)

    assert first_run == second_run


# -----------------------------------------------------------------------------
# Bounded paragraph chunking
# -----------------------------------------------------------------------------


def test_bounded_paragraph_returns_empty_list_for_whitespace_only_file() -> None:
    src_text = read_sample("blank_document.txt")

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=20,
    )

    assert chnk_list == []


def test_bounded_paragraph_preserves_paragraphs_within_limit() -> None:
    src_text = read_sample("car_engines_essay.txt")
    max_tok = 200

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 3
    assert_text_appears_in_chunks("Car Engines", chnk_list)
    assert_text_appears_in_chunks("Diesel engines", chnk_list)
    assert_text_appears_in_chunks("Fully electric cars", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_can_merge_markdown_heading_with_following_content() -> None:
    src_text = "## Site Walk\n\nThe team checked the entrance."
    max_tok = 50

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) >= 1
    assert_text_appears_in_chunks("## Site Walk", chnk_list)
    assert_text_appears_in_chunks(
        "The team checked the entrance.",
        chnk_list,
    )
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_keeps_default_markdown_behavior_without_merge_option() -> None:
    src_text = "## Site Walk\n\nThe team checked the entrance."
    max_tok = 50

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 2
    assert_text_appears_in_chunks("## Site Walk", chnk_list)
    assert_text_appears_in_chunks(
        "The team checked the entrance.",
        chnk_list,
    )
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_splits_at_sentence_boundaries() -> None:
    first_sent = (
        "The library opens early for students who need quiet study time."
    )
    second_sent = (
        "The computer lab stays open later during final exam week."
    )
    src_text = f"{first_sent} {second_sent}"

    max_tok = max(
        count_tokens(first_sent),
        count_tokens(second_sent),
    )

    assert count_tokens(src_text) > max_tok

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 2
    assert_text_appears_in_chunks(first_sent, chnk_list)
    assert_text_appears_in_chunks(second_sent, chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_packs_sentences_that_fit_together() -> None:
    first_sent = "Students reviewed the notes before class."
    second_sent = "They marked the sections that needed more work."
    third_sent = (
        "Afterward, the group wrote a much longer summary with extra details "
        "so the complete paragraph would exceed the earlier token limit."
    )
    packed_text = f"{first_sent} {second_sent}"
    src_text = f"{packed_text} {third_sent}"
    max_tok = count_tokens(packed_text)

    assert count_tokens(src_text) > max_tok

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert first_sent in chnk_list[0].chunk_text
    assert second_sent in chnk_list[0].chunk_text
    assert third_sent not in chnk_list[0].chunk_text
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_uses_word_fallback_for_long_sentence() -> None:
    src_text = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu"
    )
    max_tok = 3

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) > 1
    assert_text_appears_in_chunks("alpha", chnk_list)
    assert_text_appears_in_chunks("mu", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_handles_words_before_character_fallback() -> None:
    oversz_word = "appleorangeberrymangograpekiwipeachplum"
    leading_words = "alpha beta"
    src_text = f"{leading_words} {oversz_word}"
    max_tok = count_tokens(leading_words)

    assert count_tokens(oversz_word) > max_tok

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert "alpha" in chnk_list[0].chunk_text
    assert "beta" in chnk_list[0].chunk_text
    assert len(chnk_list) > 2
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_uses_character_fallback_for_long_word() -> None:
    src_text = "appleorangeberrymangograpekiwipeachplum"
    max_tok = 3

    assert count_tokens(src_text) > max_tok

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) > 1
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_rejects_token_limit_too_small_for_character() -> None:
    src_text = "🙂"
    max_tok = 1

    assert count_tokens(src_text) > max_tok

    with pytest.raises(
        ValidationError,
        match="Maximum token limit is too small for the selected tokenizer",
    ):
        chunk_by_bounded_paragraph(
            text=src_text,
            max_tokens=max_tok,
        )


def test_bounded_paragraph_handles_unicode_text_and_offsets() -> None:
    first_sent = "The notes include café, 東京, and several symbols."
    second_sent = "Students also recorded Δx, π, and a smile 🙂."
    src_text = f"{first_sent} {second_sent}"

    max_tok = max(
        count_tokens(first_sent),
        count_tokens(second_sent),
    )

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 2
    assert_text_appears_in_chunks("東京", chnk_list)
    assert_text_appears_in_chunks("🙂", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_preserves_leading_and_trailing_whitespace() -> None:
    src_text = (
        "\n\n"
        "First paragraph starts after leading blank lines.\n\n"
        "Second paragraph has content before trailing whitespace."
        "\n\n"
    )
    max_tok = 100

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert chnk_list[0].start_char == 0
    assert chnk_list[-1].end_char == len(src_text)
    assert_text_appears_in_chunks("First paragraph", chnk_list)
    assert_text_appears_in_chunks("Second paragraph", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_preserves_large_whitespace_gap() -> None:
    src_text = (
        "First paragraph has content."
        "\n\n\n\n\n\n\n\n\n\n"
        "Second paragraph comes after a large gap."
    )
    max_tok = 20

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 2
    assert_text_appears_in_chunks("First paragraph", chnk_list)
    assert_text_appears_in_chunks("Second paragraph", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_splits_large_trailing_whitespace() -> None:
    trail_gap = "\n" * 100
    src_text = f"A{trail_gap}"
    max_tok = 1

    assert count_tokens(trail_gap) > max_tok

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) > 2
    assert chnk_list[-1].chunk_text.isspace()
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_bounded_paragraph_rejects_invalid_token_limit() -> None:
    src_text = read_sample("data_structures_lecture_note.txt")

    with pytest.raises(
        ValidationError,
        match="Maximum tokens must be greater than 0",
    ):
        chunk_by_bounded_paragraph(
            text=src_text,
            max_tokens=0,
        )


def test_bounded_paragraph_is_deterministic() -> None:
    src_text = (
        "The first sentence contains several words for testing. "
        "The second sentence should create another bounded section. "
        "The final sentence confirms that ordering stays consistent."
    )
    max_tok = 12

    first_run = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )
    second_run = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert first_run == second_run
    assert_indexes_are_sequential(first_run)
    assert_bounded_reconstruction(src_text, first_run)
    assert_chunks_fit_token_limit(first_run, max_tok)


# -----------------------------------------------------------------------------
# Fixed-size word chunking
# -----------------------------------------------------------------------------


def test_fixed_chunking_creates_expected_overlapping_word_windows() -> None:
    src_text = read_sample("bird_watching_article.txt")
    expected_chunks = read_expected("bird_watching_fixed_12_3.json")

    chnk_list = chunk_by_fixed_words(
        src_text,
        chunk_size=12,
        overlap=3,
    )

    assert chunk_texts(chnk_list) == expected_chunks
    assert len(chnk_list) == 10

    for chnk_item in chnk_list:
        assert len(split_words(chnk_item.chunk_text)) <= 12

    assert_fixed_overlap(chnk_list, overlap=3)
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_fixed_chunking_preserves_irregular_spacing_and_offsets() -> None:
    src_text = read_sample("course_schedule_scramble.txt")
    expected_chunks = read_expected("course_schedule_fixed_8_2.json")

    chnk_list = chunk_by_fixed_words(
        src_text,
        chunk_size=8,
        overlap=2,
    )

    assert chunk_texts(chnk_list) == expected_chunks
    assert "   " in chnk_list[0].chunk_text
    assert "\t" in chnk_list[1].chunk_text
    assert "waitlist #4" in chnk_list[3].chunk_text
    assert_fixed_overlap(chnk_list, overlap=2)
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_fixed_chunking_handles_unicode_text_and_offsets() -> None:
    src_text = read_sample("unicode_japanese_notes.txt")
    expected_chunks = read_expected("unicode_japanese_fixed_7_2.json")

    chnk_list = chunk_by_fixed_words(
        src_text,
        chunk_size=7,
        overlap=2,
    )

    assert chunk_texts(chnk_list) == expected_chunks
    assert "東京" in chnk_list[1].chunk_text
    assert "Δx" in chnk_list[2].chunk_text
    assert "🙂" in chnk_list[-1].chunk_text
    assert_fixed_overlap(chnk_list, overlap=2)
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_fixed_chunking_returns_one_chunk_when_input_is_shorter_than_window() -> None:
    src_text = read_sample("data_structures_lecture_note.txt")

    chnk_list = chunk_by_fixed_words(
        src_text,
        chunk_size=50,
        overlap=10,
    )

    assert len(chnk_list) == 1
    assert chnk_list[0].chunk_index == 0
    assert chnk_list[0].chunk_text == src_text
    assert chnk_list[0].start_char == 0
    assert chnk_list[0].end_char == len(src_text)
    assert_offsets_match_source(src_text, chnk_list)


def test_fixed_chunking_returns_empty_list_for_whitespace_only_file() -> None:
    src_text = read_sample("blank_document.txt")

    chnk_list = chunk_by_fixed_words(
        src_text,
        chunk_size=8,
        overlap=2,
    )

    assert chnk_list == []


def test_fixed_chunking_rejects_zero_chunk_size() -> None:
    src_text = read_sample("bird_watching_article.txt")

    with pytest.raises(
        ValidationError,
        match="Chunk size must be greater than 0",
    ):
        chunk_by_fixed_words(
            src_text,
            chunk_size=0,
            overlap=0,
        )


def test_fixed_chunking_rejects_negative_chunk_size() -> None:
    src_text = read_sample("bird_watching_article.txt")

    with pytest.raises(
        ValidationError,
        match="Chunk size must be greater than 0",
    ):
        chunk_by_fixed_words(
            src_text,
            chunk_size=-5,
            overlap=0,
        )


def test_fixed_chunking_rejects_negative_overlap() -> None:
    src_text = read_sample("bird_watching_article.txt")

    with pytest.raises(
        ValidationError,
        match="Overlap must be greater than or equal to 0",
    ):
        chunk_by_fixed_words(
            src_text,
            chunk_size=12,
            overlap=-1,
        )


def test_fixed_chunking_rejects_overlap_equal_to_chunk_size() -> None:
    src_text = read_sample("bird_watching_article.txt")

    with pytest.raises(
        ValidationError,
        match="Overlap must be smaller than chunk size",
    ):
        chunk_by_fixed_words(
            src_text,
            chunk_size=12,
            overlap=12,
        )


def test_fixed_chunking_rejects_overlap_greater_than_chunk_size() -> None:
    src_text = read_sample("bird_watching_article.txt")

    with pytest.raises(
        ValidationError,
        match="Overlap must be smaller than chunk size",
    ):
        chunk_by_fixed_words(
            src_text,
            chunk_size=12,
            overlap=13,
        )


def test_fixed_chunking_is_deterministic_for_same_input_and_settings() -> None:
    src_text = read_sample("bird_watching_article.txt")

    first_run = chunk_by_fixed_words(
        src_text,
        chunk_size=12,
        overlap=3,
    )
    second_run = chunk_by_fixed_words(
        src_text,
        chunk_size=12,
        overlap=3,
    )

    assert first_run == second_run


# -----------------------------------------------------------------------------
# Public dispatcher function
# -----------------------------------------------------------------------------


def test_chunk_text_dispatches_to_paragraph_chunking() -> None:
    src_text = read_sample("car_engines_essay.txt")

    chnk_list = chunk_text(
        src_text,
        method="paragraph",
    )

    assert len(chnk_list) == 3
    assert chnk_list[0].chunk_text.startswith("Car Engines")
    assert chnk_list[-1].chunk_text.startswith("Hybrid cars")
    assert_offsets_match_source(src_text, chnk_list)


def test_chunk_text_can_merge_markdown_headings_for_paragraph_chunking() -> None:
    src_text = "# Notes\n\nThis paragraph should stay attached."

    chnk_list = chunk_text(
        text=src_text,
        method="paragraph",
        merge_markdown_headings=True,
    )

    assert len(chnk_list) == 1
    assert chnk_list[0].chunk_text == src_text
    assert_indexes_are_sequential(chnk_list)
    assert_offsets_match_source(src_text, chnk_list)


def test_chunk_text_dispatches_to_bounded_paragraph_chunking() -> None:
    src_text = (
        "The first sentence contains several words for testing. "
        "The second sentence should be placed in another bounded chunk."
    )
    max_tok = 12

    chnk_list = chunk_text(
        text=src_text,
        method="bounded-paragraph",
        max_tokens=max_tok,
    )

    assert len(chnk_list) > 1
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_chunk_text_can_merge_markdown_headings_for_bounded_paragraph_chunking() -> None:
    src_text = "## Site Walk\n\nThe team reviewed the front gate."
    max_tok = 50

    chnk_list = chunk_text(
        text=src_text,
        method="bounded-paragraph",
        max_tokens=max_tok,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) >= 1
    assert_text_appears_in_chunks("## Site Walk", chnk_list)
    assert_text_appears_in_chunks("front gate", chnk_list)
    assert_indexes_are_sequential(chnk_list)
    assert_bounded_reconstruction(src_text, chnk_list)
    assert_chunks_fit_token_limit(chnk_list, max_tok)


def test_chunk_text_dispatches_to_fixed_word_chunking() -> None:
    src_text = read_sample("bird_watching_article.txt")

    chnk_list = chunk_text(
        src_text,
        method="fixed",
        chunk_size=12,
        overlap=3,
    )

    assert len(chnk_list) == 10
    assert_fixed_overlap(chnk_list, overlap=3)
    assert_offsets_match_source(src_text, chnk_list)


def test_chunk_text_accepts_method_case_and_extra_spaces() -> None:
    src_text = read_sample("construction_meeting_notes.md")

    chnk_list = chunk_text(
        src_text,
        method="  PARAGRAPH  ",
    )

    assert len(chnk_list) == 5
    assert chnk_list[0].chunk_text == "# Construction Meeting Notes"
    assert_indexes_are_sequential(chnk_list)


def test_chunk_text_rejects_unknown_method() -> None:
    src_text = read_sample("car_engines_essay.txt")

    with pytest.raises(
        ValidationError,
        match="Method must be 'paragraph', 'bounded-paragraph', or 'fixed'",
    ):
        chunk_text(
            src_text,
            method="sentence",
        )