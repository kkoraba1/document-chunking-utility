import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from document_chunking_utility.paths import to_portable_path
from document_chunking_utility.validation import (
    ValidationError,
    is_empty_text,
    is_supported_file,
    validate_input_directory,
)

OUTPUT_FIELDNAMES = [
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

SUPPORTED_OUTPUT_EXTENSIONS = {".csv", ".jsonl"}


@dataclass(frozen=True)
class SourceDocument:
    source_file: str
    source_path: str
    text: str


@dataclass(frozen=True)
class LoadResult:
    documents: list[SourceDocument]
    processed_files: int
    unsupported_files_skipped: int
    empty_files_skipped: int
    read_error_files_skipped: int = 0


def discover_files(input_dir: str | Path) -> list[Path]:
    """Find all files inside the input directory."""
    root = validate_input_directory(input_dir)
    found_files: list[Path] = []

    for file_path in root.rglob("*"):
        if file_path.is_file():
            found_files.append(file_path)

    # Sort using portable relative paths so the order stays predictable.
    found_files.sort(
        key=lambda file_path: to_portable_path(
            file_path.relative_to(root)
        ).lower()
    )

    return found_files


def read_text_file(path: str | Path) -> str:
    """Read a text file using UTF-8 encoding."""
    file_path = Path(path)

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as text_file:
            return text_file.read()

    except UnicodeDecodeError as exc:
        raise ValidationError(
            f"Could not decode supported file as UTF-8: {file_path}"
        ) from exc

    except OSError as exc:
        raise ValidationError(
            f"Could not read supported file: {file_path}"
        ) from exc


def load_documents(input_dir: str | Path) -> LoadResult:
    """Load supported, non-empty documents from the input directory."""
    root = validate_input_directory(input_dir)
    found_files = discover_files(root)

    documents: list[SourceDocument] = []
    unsupported_skips = 0
    empty_skips = 0
    read_error_skips = 0

    # Check each discovered file before adding it as a document.
    for file_path in found_files:
        if not is_supported_file(file_path):
            unsupported_skips += 1
            continue

        try:
            text = read_text_file(file_path)

        except ValidationError:
            read_error_skips += 1
            continue

        if is_empty_text(text):
            empty_skips += 1
            continue

        documents.append(
            SourceDocument(
                source_file=file_path.name,
                source_path=to_portable_path(
                    file_path.relative_to(root)
                ),
                text=text,
            )
        )

    return LoadResult(
        documents=documents,
        processed_files=len(documents),
        unsupported_files_skipped=unsupported_skips,
        empty_files_skipped=empty_skips,
        read_error_files_skipped=read_error_skips,
    )


def write_records(
    records: Sequence[Mapping[str, Any] | object],
    output_path: str | Path,
) -> Path:
    """Write chunk records based on the selected output file type."""
    path = validate_output_path(output_path)

    if path.suffix.lower() == ".csv":
        return write_csv(
            records=records,
            output_path=path,
        )

    if path.suffix.lower() == ".jsonl":
        return write_jsonl(
            records=records,
            output_path=path,
        )

    raise ValidationError(
        f"Unsupported output file type: {path.suffix}"
    )


def write_csv(
    records: Sequence[Mapping[str, Any] | object],
    output_path: str | Path,
) -> Path:
    """Write chunk records to a CSV file."""
    path = validate_output_path(
        output_path,
        expected_suffix=".csv",
    )

    # Create the output folder if it does not already exist.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=OUTPUT_FIELDNAMES,
        )

        writer.writeheader()

        for record in records:
            ordered_record = _schema_ordered_record(record)
            writer.writerow(ordered_record)

    return path


def write_jsonl(
    records: Sequence[Mapping[str, Any] | object],
    output_path: str | Path,
) -> Path:
    """Write one JSON object per chunk record."""
    path = validate_output_path(
        output_path,
        expected_suffix=".jsonl",
    )

    # Create the output folder if it does not already exist.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            ordered_record = _schema_ordered_record(record)

            json.dump(
                ordered_record,
                jsonl_file,
                ensure_ascii=False,
            )

            jsonl_file.write("\n")

    return path


def validate_output_path(
    output_path: str | Path,
    expected_suffix: str | None = None,
) -> Path:
    """Check that an output path uses a supported file extension."""
    path = Path(output_path)
    suffix = path.suffix.lower()

    if expected_suffix is not None and suffix != expected_suffix:
        raise ValidationError(
            f"Output path must end with {expected_suffix}: {path}"
        )

    if suffix not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported_exts = ", ".join(
            sorted(SUPPORTED_OUTPUT_EXTENSIONS)
        )

        raise ValidationError(
            f"Unsupported output file type: {suffix or '<none>'}. "
            f"Supported extensions are: {supported_exts}"
        )

    return path


def _schema_ordered_record(
    record: Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Put a record's values in the expected output field order."""
    raw_record = _record_to_dict(record)
    ordered_record: dict[str, Any] = {}

    for field_name in OUTPUT_FIELDNAMES:
        ordered_record[field_name] = raw_record.get(
            field_name,
            "",
        )

    return ordered_record


def _record_to_dict(
    record: Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Convert a supported record into a regular dictionary."""
    if isinstance(record, Mapping):
        return dict(record)

    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)

    raise TypeError(
        "Chunk records must be dictionaries, mappings, "
        "or dataclass instances."
    )