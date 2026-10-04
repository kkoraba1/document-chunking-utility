"""Tests for cross-platform path formatting helpers."""

from pathlib import Path

from document_chunking_utility.paths import to_portable_path


def test_to_portable_path_converts_windows_separators() -> None:
    raw_path = r"semester_notes\week_04\lecture_summary.md"

    port_path = to_portable_path(raw_path)

    assert port_path == "semester_notes/week_04/lecture_summary.md"


def test_to_portable_path_preserves_posix_separators() -> None:
    raw_path = "semester_notes/week_04/lecture_summary.md"

    port_path = to_portable_path(raw_path)

    assert port_path == "semester_notes/week_04/lecture_summary.md"


def test_to_portable_path_accepts_path_objects() -> None:
    raw_path = (
        Path("coursework")
        / "data_structures"
        / "lecture_note.txt"
    )

    port_path = to_portable_path(raw_path)

    assert port_path == "coursework/data_structures/lecture_note.txt"


def test_to_portable_path_handles_mixed_separators() -> None:
    raw_path = r"campus_projects\construction/site_notes.md"

    port_path = to_portable_path(raw_path)

    assert port_path == "campus_projects/construction/site_notes.md"


def test_to_portable_path_preserves_spaces_and_punctuation() -> None:
    raw_path = (
        r"office hours\Spring 2026\Professor Schedule - final.txt"
    )

    port_path = to_portable_path(raw_path)

    assert (
        port_path
        == "office hours/Spring 2026/Professor Schedule - final.txt"
    )


def test_to_portable_path_preserves_unicode_characters() -> None:
    raw_path = r"language_notes\日本語\東京_field_note.txt"

    port_path = to_portable_path(raw_path)

    assert port_path == "language_notes/日本語/東京_field_note.txt"


def test_to_portable_path_preserves_relative_path_markers() -> None:
    raw_path = r"..\archive\week_01\notes.md"

    port_path = to_portable_path(raw_path)

    assert port_path == "../archive/week_01/notes.md"


def test_to_portable_path_returns_file_name_when_no_directory_is_present() -> None:
    raw_path = "car_engines_essay.txt"

    port_path = to_portable_path(raw_path)

    assert port_path == "car_engines_essay.txt"