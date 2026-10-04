"""Tests for configuration-aware chunk identity."""

import pytest

from document_chunking_utility.metadata import build_chunk_id


def test_paragraph_chunk_id_includes_method() -> None:
    chnk_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="paragraph",
    )

    assert chnk_id == "notes_txt_83348d28_paragraph_000"


def test_fixed_chunk_id_includes_chunk_size_and_overlap() -> None:
    chnk_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=200,
        overlap=40,
    )

    assert chnk_id == "notes_txt_83348d28_fixed_cs200_ov40_000"


def test_bounded_paragraph_chunk_id_includes_token_settings() -> None:
    chnk_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="bounded-paragraph",
        max_tokens=512,
        tokenizer="cl100k_base",
    )

    assert chnk_id == (
        "notes_txt_83348d28_"
        "bounded_paragraph_mt512_tk_cl100k_base_000"
    )


def test_chunk_id_distinguishes_different_methods() -> None:
    para_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="paragraph",
    )

    fixed_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=200,
        overlap=40,
    )

    assert para_id != fixed_id


def test_chunk_id_distinguishes_fixed_chunk_size() -> None:
    first_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=100,
        overlap=20,
    )

    second_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=200,
        overlap=20,
    )

    assert first_id != second_id


def test_chunk_id_distinguishes_fixed_overlap() -> None:
    first_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=200,
        overlap=20,
    )

    second_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="fixed",
        chunk_size=200,
        overlap=40,
    )

    assert first_id != second_id


def test_chunk_id_distinguishes_bounded_max_tokens() -> None:
    first_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="bounded-paragraph",
        max_tokens=256,
        tokenizer="cl100k_base",
    )

    second_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="bounded-paragraph",
        max_tokens=512,
        tokenizer="cl100k_base",
    )

    assert first_id != second_id


def test_chunk_id_distinguishes_bounded_tokenizer() -> None:
    first_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="bounded-paragraph",
        max_tokens=512,
        tokenizer="cl100k_base",
    )

    second_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=0,
        method="bounded-paragraph",
        max_tokens=512,
        tokenizer="p50k_base",
    )

    assert first_id != second_id


def test_chunk_id_is_deterministic_for_same_configuration() -> None:
    first_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=2,
        method="fixed",
        chunk_size=200,
        overlap=40,
    )

    second_id = build_chunk_id(
        source_path="notes.txt",
        chunk_index=2,
        method="fixed",
        chunk_size=200,
        overlap=40,
    )

    assert first_id == second_id


def test_fixed_chunk_id_requires_chunk_size() -> None:
    with pytest.raises(ValueError, match="Chunk size is required"):
        build_chunk_id(
            source_path="notes.txt",
            chunk_index=0,
            method="fixed",
            overlap=40,
        )


def test_fixed_chunk_id_requires_overlap() -> None:
    with pytest.raises(ValueError, match="Overlap is required"):
        build_chunk_id(
            source_path="notes.txt",
            chunk_index=0,
            method="fixed",
            chunk_size=200,
        )


def test_bounded_paragraph_chunk_id_requires_max_tokens() -> None:
    with pytest.raises(ValueError, match="Maximum tokens are required"):
        build_chunk_id(
            source_path="notes.txt",
            chunk_index=0,
            method="bounded-paragraph",
            tokenizer="cl100k_base",
        )


def test_bounded_paragraph_chunk_id_requires_tokenizer() -> None:
    with pytest.raises(ValueError, match="Tokenizer is required"):
        build_chunk_id(
            source_path="notes.txt",
            chunk_index=0,
            method="bounded-paragraph",
            max_tokens=512,
        )