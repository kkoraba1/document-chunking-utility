"""Tests for exact bounded paragraph source reconstruction."""

from document_chunking_utility.chunkers import chunk_by_bounded_paragraph
from document_chunking_utility.tokenizer import count_tokens


def assert_chunk_offsets_match(src_text, chnk_list):
    """Check that each chunk matches its exact slice in the source text."""
    for chnk_item in chnk_list:
        assert chnk_item.start_char < chnk_item.end_char
        assert (
            src_text[chnk_item.start_char:chnk_item.end_char]
            == chnk_item.chunk_text
        )


def assert_chunks_stay_contiguous(src_text, chnk_list):
    """Check that chunks cover the source text in order without gaps."""
    if not chnk_list:
        assert src_text.strip() == ""
        return

    next_start = 0

    for chnk_item in chnk_list:
        assert chnk_item.start_char == next_start
        next_start = chnk_item.end_char

    assert next_start == len(src_text)


def assert_chunks_rebuild_source(src_text, chnk_list):
    """Check that joining all chunk text rebuilds the full source."""
    rebuilt_text = ""

    for chnk_item in chnk_list:
        rebuilt_text += chnk_item.chunk_text

    assert rebuilt_text == src_text


def assert_chunks_fit_limit(chnk_list, max_tok):
    """Check that every chunk stays within the token limit."""
    for chnk_item in chnk_list:
        tok_count = count_tokens(chnk_item.chunk_text)
        assert tok_count <= max_tok


def assert_full_reconstruction(src_text, chnk_list):
    """Run all source reconstruction checks."""
    assert_chunk_offsets_match(src_text, chnk_list)
    assert_chunks_stay_contiguous(src_text, chnk_list)
    assert_chunks_rebuild_source(src_text, chnk_list)


def assert_text_exists(expected_text, chnk_list):
    """Check that important text appears somewhere in the chunks."""
    text_found = False

    for chnk_item in chnk_list:
        if expected_text in chnk_item.chunk_text:
            text_found = True

    assert text_found


def test_bounded_reconstruction_preserves_unsplit_paragraphs() -> None:
    src_text = (
        "First paragraph with normal project notes.\n\n"
        "Second paragraph with extra details.\n\n"
        "Third paragraph closes the document."
    )
    max_tok = 100

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 3
    assert_text_exists(
        "First paragraph with normal project notes.",
        chnk_list,
    )
    assert_text_exists(
        "Second paragraph with extra details.",
        chnk_list,
    )
    assert_text_exists(
        "Third paragraph closes the document.",
        chnk_list,
    )
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_sentence_split_chunks() -> None:
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

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) >= 2
    assert_text_exists(first_sent, chnk_list)
    assert_text_exists(second_sent, chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_word_fallback_chunks() -> None:
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
    assert_text_exists("alpha", chnk_list)
    assert_text_exists("mu", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_character_fallback_chunks() -> None:
    src_text = "supercalifragilisticexpialidocious"
    max_tok = 3

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert len(chnk_list) > 1
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_markdown_heading_merge() -> None:
    src_text = (
        "# Project Notes\n\n"
        "## Site Walk\n\n"
        "The team checked the north entrance and listed repair tasks."
    )
    max_tok = 100

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
        merge_markdown_headings=True,
    )

    assert len(chnk_list) >= 1
    assert_text_exists("# Project Notes", chnk_list)
    assert_text_exists("## Site Walk", chnk_list)
    assert_text_exists("north entrance", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_markdown_without_heading_merge() -> None:
    src_text = (
        "# Project Notes\n\n"
        "The team checked the north entrance and listed repair tasks."
    )
    max_tok = 100

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
        merge_markdown_headings=False,
    )

    assert len(chnk_list) >= 2
    assert_text_exists("# Project Notes", chnk_list)
    assert_text_exists("north entrance", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_crlf_line_endings() -> None:
    src_text = (
        "Office hours are Monday morning.\r\n"
        "Students should bring draft questions.\r\n\r\n"
        "The professor will review project outlines.\r\n"
        "Follow-up notes should be submitted by Friday."
    )
    max_tok = 100

    chnk_list = chunk_by_bounded_paragraph(
        text=src_text,
        max_tokens=max_tok,
    )

    assert "\r\n" in src_text
    assert len(chnk_list) >= 2
    assert_text_exists("Office hours", chnk_list)
    assert_text_exists("Follow-up notes", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_unicode_text() -> None:
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
    assert_text_exists("東京", chnk_list)
    assert_text_exists("🙂", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_preserves_leading_and_trailing_whitespace() -> None:
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

    assert len(chnk_list) >= 2
    assert chnk_list[0].start_char == 0
    assert chnk_list[-1].end_char == len(src_text)
    assert_text_exists("First paragraph", chnk_list)
    assert_text_exists("Second paragraph", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)


def test_bounded_reconstruction_handles_large_whitespace_gap() -> None:
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
    assert_text_exists("First paragraph", chnk_list)
    assert_text_exists("Second paragraph", chnk_list)
    assert_full_reconstruction(src_text, chnk_list)
    assert_chunks_fit_limit(chnk_list, max_tok)