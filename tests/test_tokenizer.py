"""Tests for token-counting helpers."""

import pytest
from tiktoken import Encoding

from document_chunking_utility.tokenizer import (
    DEFAULT_TOKENIZER,
    count_tokens,
    load_tokenizer,
)
from document_chunking_utility.validation import ValidationError


def test_load_tokenizer_uses_default_encoding() -> None:
    tok_item = load_tokenizer()

    assert isinstance(tok_item, Encoding)
    assert tok_item.name == DEFAULT_TOKENIZER


def test_load_tokenizer_accepts_named_encoding() -> None:
    tok_item = load_tokenizer("cl100k_base")

    assert tok_item.name == "cl100k_base"


def test_load_tokenizer_removes_extra_spaces() -> None:
    tok_item = load_tokenizer("  cl100k_base  ")

    assert tok_item.name == "cl100k_base"


def test_load_tokenizer_rejects_empty_name() -> None:
    with pytest.raises(
        ValidationError,
        match="Tokenizer name cannot be empty",
    ):
        load_tokenizer("   ")


def test_load_tokenizer_rejects_unknown_encoding() -> None:
    with pytest.raises(
        ValidationError,
        match="Unknown tokenizer encoding",
    ):
        load_tokenizer("not_a_real_tokenizer")


def test_count_tokens_returns_integer_count() -> None:
    tok_count = count_tokens("This is a short paragraph.")

    assert isinstance(tok_count, int)
    assert tok_count > 0


def test_count_tokens_returns_zero_for_empty_text() -> None:
    tok_count = count_tokens("")

    assert tok_count == 0


def test_count_tokens_handles_unicode_text() -> None:
    src_text = "Notes from Tokyo: 東京 猫 ありがとう 🙂"

    tok_count = count_tokens(src_text)

    assert tok_count > 0


def test_count_tokens_is_deterministic() -> None:
    src_text = "The same text should always produce the same token count."

    first_count = count_tokens(src_text)
    second_count = count_tokens(src_text)

    assert first_count == second_count