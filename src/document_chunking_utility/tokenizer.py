"""Token-counting helpers for bounded paragraph chunking."""

import tiktoken
from tiktoken import Encoding

from document_chunking_utility.validation import ValidationError

DEFAULT_TOKENIZER = "cl100k_base"


def load_tokenizer(
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> Encoding:
    """Load a tiktoken encoding using its name."""
    tok_name = tokenizer_name.strip()

    if not tok_name:
        raise ValidationError(
            "Tokenizer name cannot be empty."
        )

    try:
        return tiktoken.get_encoding(tok_name)

    except ValueError as exc:
        raise ValidationError(
            f"Unknown tokenizer encoding: {tok_name}"
        ) from exc


def count_tokens(
    text: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> int:
    """Count how many tokens are in a text string."""
    tokenizer = load_tokenizer(tokenizer_name)
    token_ids = tokenizer.encode(text)

    return len(token_ids)