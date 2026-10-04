"""Tests for validation helpers."""

from pathlib import Path

import pytest

from document_chunking_utility.validation import (
    ValidationError,
    is_empty_text,
    is_supported_file,
    validate_chunk_settings,
    validate_input_directory,
    validate_max_tokens,
    validate_non_empty_text,
    validate_supported_file,
)

# -----------------------------------------------------------------------------
# Input directory validation
# -----------------------------------------------------------------------------


def test_validate_input_directory_accepts_existing_directory(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "student_notes"
    input_dir.mkdir()

    valid_path = validate_input_directory(input_dir)

    assert valid_path == input_dir.resolve()


def test_validate_input_directory_accepts_string_paths(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "semester_documents"
    input_dir.mkdir()

    valid_path = validate_input_directory(str(input_dir))

    assert valid_path == input_dir.resolve()


def test_validate_input_directory_rejects_missing_path(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing_course_folder"

    with pytest.raises(
        ValidationError,
        match="Input path does not exist",
    ):
        validate_input_directory(missing_path)


def test_validate_input_directory_rejects_file_path(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "office_hour_timings.txt"

    file_path.write_text(
        "Dr. Allen: Tuesday 2-4, room 315. "
        "Bring the worksheet if possible.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="Input path is not a directory",
    ):
        validate_input_directory(file_path)


# -----------------------------------------------------------------------------
# Supported file validation
# -----------------------------------------------------------------------------


def test_is_supported_file_accepts_text_and_markdown_documents() -> None:
    assert is_supported_file("car_engines_essay.txt")
    assert is_supported_file("construction_meeting_notes.md")
    assert is_supported_file(
        Path("week_04") / "lecture_notes.txt"
    )


def test_is_supported_file_is_case_insensitive() -> None:
    assert is_supported_file(
        "DATA_STRUCTURES_LECTURE_NOTE.TXT"
    )
    assert is_supported_file(
        "CONSTRUCTION_MEETING_NOTES.MD"
    )


def test_is_supported_file_rejects_unsupported_extensions() -> None:
    assert not is_supported_file("bird_photo.jpeg")
    assert not is_supported_file("construction_budget.xlsx")
    assert not is_supported_file("meeting_recording.mp3")
    assert not is_supported_file("field_notes.pdf")
    assert not is_supported_file("course_schedule")


def test_validate_supported_file_returns_path_for_supported_file() -> None:
    valid_path = validate_supported_file(
        "notes/week_03/data_structures.md"
    )

    assert valid_path == Path(
        "notes/week_03/data_structures.md"
    )


def test_validate_supported_file_accepts_uppercase_supported_extension() -> None:
    valid_path = validate_supported_file(
        "Professor_Office_Hours.TXT"
    )

    assert valid_path == Path(
        "Professor_Office_Hours.TXT"
    )


def test_validate_supported_file_rejects_unsupported_file() -> None:
    with pytest.raises(
        ValidationError,
        match="Unsupported file type",
    ):
        validate_supported_file("campus_map.png")


def test_validate_supported_file_error_lists_supported_extensions() -> None:
    with pytest.raises(ValidationError) as err_info:
        validate_supported_file("bird_sightings.csv")

    err_text = str(err_info.value)

    assert "Unsupported file type: .csv" in err_text
    assert ".md" in err_text
    assert ".txt" in err_text


# -----------------------------------------------------------------------------
# Empty text validation
# -----------------------------------------------------------------------------


def test_is_empty_text_detects_empty_or_whitespace_only_text() -> None:
    assert is_empty_text("")
    assert is_empty_text("   ")
    assert is_empty_text("\n\t  \n")
    assert is_empty_text(" \r\n\t \r\n ")


def test_is_empty_text_rejects_real_document_content() -> None:
    assert not is_empty_text(
        "Car engines turn fuel into motion by using combustion."
    )
    assert not is_empty_text(
        "  Office hours moved to Thursday after lunch.  "
    )
    assert not is_empty_text(
        "東京 notes with symbols like Δx and 🙂 are still text."
    )


def test_validate_non_empty_text_accepts_document_content() -> None:
    validate_non_empty_text(
        (
            "Professor Lee said the midterm review will mostly "
            "cover stacks and queues."
        ),
        "data_structures_lecture_note.txt",
    )


def test_validate_non_empty_text_rejects_blank_document_text() -> None:
    with pytest.raises(
        ValidationError,
        match="Empty document skipped",
    ):
        validate_non_empty_text(
            "   \n\t  \n",
            "blank_document.txt",
        )


def test_validate_non_empty_text_error_includes_source_path() -> None:
    with pytest.raises(ValidationError) as err_info:
        validate_non_empty_text(
            "\n\n   ",
            "week_02/empty_office_hours.md",
        )

    assert "week_02/empty_office_hours.md" in str(err_info.value)


# -----------------------------------------------------------------------------
# Fixed-size chunk setting validation
# -----------------------------------------------------------------------------


def test_validate_chunk_settings_accepts_typical_fixed_chunk_settings() -> None:
    validate_chunk_settings(
        chunk_size=12,
        overlap=3,
    )
    validate_chunk_settings(
        chunk_size=50,
        overlap=10,
    )
    validate_chunk_settings(
        chunk_size=200,
        overlap=40,
    )


def test_validate_chunk_settings_accepts_zero_overlap() -> None:
    validate_chunk_settings(
        chunk_size=25,
        overlap=0,
    )


def test_validate_chunk_settings_accepts_overlap_one_less_than_chunk_size() -> None:
    validate_chunk_settings(
        chunk_size=8,
        overlap=7,
    )


def test_validate_chunk_settings_rejects_zero_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="Chunk size must be greater than 0",
    ):
        validate_chunk_settings(
            chunk_size=0,
            overlap=0,
        )


def test_validate_chunk_settings_rejects_negative_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="Chunk size must be greater than 0",
    ):
        validate_chunk_settings(
            chunk_size=-10,
            overlap=0,
        )


def test_validate_chunk_settings_rejects_negative_overlap() -> None:
    with pytest.raises(
        ValidationError,
        match="Overlap must be greater than or equal to 0",
    ):
        validate_chunk_settings(
            chunk_size=100,
            overlap=-1,
        )


def test_validate_chunk_settings_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="Overlap must be smaller than chunk size",
    ):
        validate_chunk_settings(
            chunk_size=100,
            overlap=100,
        )


def test_validate_chunk_settings_rejects_overlap_greater_than_chunk_size() -> None:
    with pytest.raises(
        ValidationError,
        match="Overlap must be smaller than chunk size",
    ):
        validate_chunk_settings(
            chunk_size=100,
            overlap=120,
        )


# -----------------------------------------------------------------------------
# Validation error type
# -----------------------------------------------------------------------------


def test_validation_error_is_a_value_error() -> None:
    valid_err = ValidationError(
        "The input folder is not usable."
    )

    assert isinstance(valid_err, ValueError)


# -----------------------------------------------------------------------------
# Maximum token validation
# -----------------------------------------------------------------------------


def test_validate_max_tokens_accepts_positive_limit() -> None:
    validate_max_tokens(512)


def test_validate_max_tokens_accepts_limit_of_one() -> None:
    validate_max_tokens(1)


def test_validate_max_tokens_rejects_zero() -> None:
    with pytest.raises(
        ValidationError,
        match="Maximum tokens must be greater than 0",
    ):
        validate_max_tokens(0)


def test_validate_max_tokens_rejects_negative_limit() -> None:
    with pytest.raises(
        ValidationError,
        match="Maximum tokens must be greater than 0",
    ):
        validate_max_tokens(-100)