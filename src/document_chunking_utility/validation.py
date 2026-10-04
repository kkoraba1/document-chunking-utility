from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


class ValidationError(ValueError):
    """Custom error used for invalid utility input."""



def validate_input_directory(input_dir: str | Path) -> Path:
    """Check that the input path exists and is a directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValidationError(
            f"Input path does not exist: {input_path}"
        )

    if not input_path.is_dir():
        raise ValidationError(
            f"Input path is not a directory: {input_path}"
        )

    return input_path.resolve()


def is_supported_file(path: str | Path) -> bool:
    """Return True when the file has a supported extension."""
    file_path = Path(path)

    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def validate_supported_file(path: str | Path) -> Path:
    """Check that a file uses a supported extension."""
    file_path = Path(path)

    if not is_supported_file(file_path):
        supported_exts = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValidationError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported extensions are: {supported_exts}"
        )

    return file_path


def is_empty_text(text: str) -> bool:
    """Return True when text contains no visible content."""
    return not text.strip()


def validate_non_empty_text(
    text: str,
    source_path: str | Path,
) -> None:
    """Check that a document contains visible text."""
    if is_empty_text(text):
        raise ValidationError(
            f"Empty document skipped: {source_path}"
        )


def validate_chunk_settings(
    chunk_size: int,
    overlap: int,
) -> None:
    """Check the fixed-size chunk settings."""
    if chunk_size <= 0:
        raise ValidationError(
            "Chunk size must be greater than 0."
        )

    if overlap < 0:
        raise ValidationError(
            "Overlap must be greater than or equal to 0."
        )

    if overlap >= chunk_size:
        raise ValidationError(
            "Overlap must be smaller than chunk size."
        )


def validate_max_tokens(max_tokens: int) -> None:
    """Check the maximum token limit for bounded chunks."""
    if max_tokens <= 0:
        raise ValidationError(
            "Maximum tokens must be greater than 0."
        )