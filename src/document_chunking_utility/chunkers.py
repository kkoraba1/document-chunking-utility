"""Text chunking helpers for the Document Chunking Utility."""

import re
from dataclasses import dataclass

from document_chunking_utility.tokenizer import (
    DEFAULT_TOKENIZER,
    count_tokens,
)
from document_chunking_utility.validation import (
    ValidationError,
    validate_chunk_settings,
    validate_max_tokens,
)


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    chunk_text: str
    start_char: int
    end_char: int


def _validate_chunk_offsets(text: str, chunks: list[TextChunk]) -> None:
    """Check that each chunk matches its location in the source text."""
    for chunk in chunks:
        if chunk.start_char < 0:
            raise ValidationError("Chunk start offset cannot be negative.")

        if chunk.end_char > len(text):
            raise ValidationError("Chunk end offset cannot exceed source length.")

        if chunk.start_char >= chunk.end_char:
            raise ValidationError("Chunk offsets must describe a non-empty span.")

        src_slice = text[chunk.start_char:chunk.end_char]

        if src_slice != chunk.chunk_text:
            raise ValidationError(
                "Chunk offsets do not reconstruct the original source text."
            )


def _validate_span_order(
    text: str,
    spans: list[tuple[int, int]],
) -> None:
    """Check that character spans are valid and do not overlap."""
    prev_end: int | None = None

    for start_char, end_char in spans:
        if start_char < 0:
            raise ValidationError("Chunk start offset cannot be negative.")

        if end_char > len(text):
            raise ValidationError("Chunk end offset cannot exceed source length.")

        if start_char >= end_char:
            raise ValidationError("Chunk offsets must describe a non-empty span.")

        if prev_end is not None and start_char < prev_end:
            raise ValidationError("Chunk spans must not overlap.")

        prev_end = end_char


def _validate_bounded_chunks_are_contiguous(
    text: str,
    chunks: list[TextChunk],
) -> None:
    """Check that bounded chunks cover the full source text in order."""
    if not chunks:
        if text.strip():
            raise ValidationError(
                "Bounded paragraph chunks do not cover the source text."
            )

        return

    expect_start = 0

    for chunk in chunks:
        if chunk.start_char != expect_start:
            raise ValidationError(
                "Bounded paragraph chunks must be contiguous."
            )

        expect_start = chunk.end_char

    if expect_start != len(text):
        raise ValidationError(
            "Bounded paragraph chunks do not cover the source text."
        )


def _validate_full_text_reconstruction(
    text: str,
    chunks: list[TextChunk],
) -> None:
    """Check that bounded chunks rebuild the original text."""
    rebuilt_text = ""

    for chunk in chunks:
        rebuilt_text += chunk.chunk_text

    if rebuilt_text != text:
        raise ValidationError(
            "Bounded paragraph chunks do not reconstruct the source text."
        )


def _validate_bounded_token_limit(
    chunks: list[TextChunk],
    max_tokens: int,
    tokenizer_name: str,
) -> None:
    """Check that every bounded chunk stays within the token limit."""
    for chunk in chunks:
        tok_count = count_tokens(
            chunk.chunk_text,
            tokenizer_name=tokenizer_name,
        )

        if tok_count > max_tokens:
            raise ValidationError(
                "Bounded paragraph chunk exceeds the maximum token limit."
            )


def _split_gap_span(
    text: str,
    start_char: int,
    end_char: int,
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[int, int]]:
    """Split a whitespace gap only when it is too large for one chunk."""
    if start_char >= end_char:
        return []

    gap_text = text[start_char:end_char]
    gap_tok_count = count_tokens(
        gap_text,
        tokenizer_name=tokenizer_name,
    )

    if gap_tok_count <= max_tokens:
        return [(start_char, end_char)]

    return _split_span_by_characters(
        text=text,
        start_char=start_char,
        end_char=end_char,
        max_tokens=max_tokens,
        tokenizer_name=tokenizer_name,
    )


def _can_use_span(
    text: str,
    start_char: int,
    end_char: int,
    max_tokens: int,
    tokenizer_name: str,
) -> bool:
    """Return True if a source span fits inside the token limit."""
    if start_char >= end_char:
        return False

    span_text = text[start_char:end_char]
    tok_count = count_tokens(
        span_text,
        tokenizer_name=tokenizer_name,
    )

    return tok_count <= max_tokens


def _make_bounded_spans_contiguous(
    text: str,
    spans: list[tuple[int, int]],
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[int, int]]:
    """Fill gaps between bounded spans so the full text can be rebuilt."""
    if not spans:
        return []

    _validate_span_order(text=text, spans=spans)

    contig_spans: list[tuple[int, int]] = []
    expect_start = 0

    for content_start, content_end in spans:
        curr_start = content_start

        # Handle skipped spaces or blank lines before this content.
        if expect_start < content_start:
            gap_start = expect_start
            gap_end = content_start
            gap_used = False

            # Try attaching the gap to the previous chunk first.
            if contig_spans:
                prev_start, _prev_end = contig_spans[-1]

                if _can_use_span(
                    text=text,
                    start_char=prev_start,
                    end_char=gap_end,
                    max_tokens=max_tokens,
                    tokenizer_name=tokenizer_name,
                ):
                    contig_spans[-1] = (prev_start, gap_end)
                    gap_used = True

            # Otherwise try attaching the gap to the current chunk.
            if not gap_used and _can_use_span(
                text=text,
                start_char=gap_start,
                end_char=content_end,
                max_tokens=max_tokens,
                tokenizer_name=tokenizer_name,
            ):
                curr_start = gap_start
                gap_used = True

            # Keep the gap separate if neither chunk can contain it.
            if not gap_used:
                gap_spans = _split_gap_span(
                    text=text,
                    start_char=gap_start,
                    end_char=gap_end,
                    max_tokens=max_tokens,
                    tokenizer_name=tokenizer_name,
                )
                contig_spans.extend(gap_spans)

        contig_spans.append((curr_start, content_end))
        expect_start = content_end

    # Handle any whitespace left after the final content span.
    if expect_start < len(text):
        trail_start = expect_start
        trail_end = len(text)
        trail_used = False

        if contig_spans:
            prev_start, _prev_end = contig_spans[-1]

            if _can_use_span(
                text=text,
                start_char=prev_start,
                end_char=trail_end,
                max_tokens=max_tokens,
                tokenizer_name=tokenizer_name,
            ):
                contig_spans[-1] = (prev_start, trail_end)
                trail_used = True

        if not trail_used:
            trail_spans = _split_gap_span(
                text=text,
                start_char=trail_start,
                end_char=trail_end,
                max_tokens=max_tokens,
                tokenizer_name=tokenizer_name,
            )
            contig_spans.extend(trail_spans)

    return contig_spans


def _is_short_markdown_heading(chunk_text: str) -> bool:
    """Return True when a chunk contains only a short Markdown heading."""
    lines = chunk_text.splitlines()

    if len(lines) != 1:
        return False

    heading_line = lines[0].strip()

    if not heading_line.startswith("#"):
        return False

    heading_lvl = 0

    # Count the opening Markdown heading markers.
    for char in heading_line:
        if char == "#":
            heading_lvl += 1
        else:
            break

    if heading_lvl < 1 or heading_lvl > 6:
        return False

    if heading_lvl >= len(heading_line):
        return False

    # A space is required so text such as "#tag" is not a heading.
    if heading_line[heading_lvl] not in (" ", "\t"):
        return False

    heading_text = heading_line[heading_lvl:].strip()

    # Remove optional closing markers from headings such as "## Notes ##".
    while heading_text.endswith("#"):
        heading_text = heading_text[:-1].strip()

    if not heading_text:
        return False

    if len(heading_text) > 120:
        return False

    heading_words = heading_text.split()

    return len(heading_words) <= 12


def _renumber_chunk(chunk: TextChunk, new_index: int) -> TextChunk:
    """Create a copy of a chunk using a new chunk index."""
    return TextChunk(
        chunk_index=new_index,
        chunk_text=chunk.chunk_text,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
    )


def _merge_isolated_markdown_headings(
    text: str,
    chunks: list[TextChunk],
) -> list[TextChunk]:
    """Attach isolated Markdown headings to the next content chunk."""
    merged_chunks: list[TextChunk] = []
    pending_heads: list[TextChunk] = []

    for chunk in chunks:
        is_heading = _is_short_markdown_heading(chunk.chunk_text)

        # Save headings until normal content appears after them.
        if is_heading:
            pending_heads.append(chunk)
            continue

        if pending_heads:
            merge_start = pending_heads[0].start_char
            merge_end = chunk.end_char

            merged_chunks.append(
                TextChunk(
                    chunk_index=len(merged_chunks),
                    chunk_text=text[merge_start:merge_end],
                    start_char=merge_start,
                    end_char=merge_end,
                )
            )

            pending_heads = []
            continue

        merged_chunks.append(
            _renumber_chunk(
                chunk=chunk,
                new_index=len(merged_chunks),
            )
        )

    # Keep headings that appear at the end without content after them.
    for heading in pending_heads:
        merged_chunks.append(
            _renumber_chunk(
                chunk=heading,
                new_index=len(merged_chunks),
            )
        )

    return merged_chunks


def chunk_by_paragraph(
    text: str,
    merge_markdown_headings: bool = False,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    para_start: int | None = None
    para_end: int | None = None
    curr_offset = 0

    # Read each line and group non-empty lines into paragraphs.
    for line in text.splitlines(keepends=True):
        line_start = curr_offset
        line_end = curr_offset + len(line)

        if line.strip():
            if para_start is None:
                para_start = line_start

            para_end = _line_content_end(line_start, line)

        else:
            if para_start is not None and para_end is not None:
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        chunk_text=text[para_start:para_end],
                        start_char=para_start,
                        end_char=para_end,
                    )
                )

                para_start = None
                para_end = None

        curr_offset = line_end

    if para_start is not None and para_end is not None:
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                chunk_text=text[para_start:para_end],
                start_char=para_start,
                end_char=para_end,
            )
        )

    if merge_markdown_headings:
        chunks = _merge_isolated_markdown_headings(
            text=text,
            chunks=chunks,
        )

    _validate_chunk_offsets(text=text, chunks=chunks)

    return chunks


def chunk_by_bounded_paragraph(
    text: str,
    max_tokens: int,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    merge_markdown_headings: bool = False,
) -> list[TextChunk]:
    """Keep normal paragraphs while splitting ones that are too large."""
    validate_max_tokens(max_tokens)

    para_chunks = chunk_by_paragraph(
        text=text,
        merge_markdown_headings=merge_markdown_headings,
    )

    chunk_spans: list[tuple[int, int]] = []

    for paragraph in para_chunks:
        para_tok_count = count_tokens(
            paragraph.chunk_text,
            tokenizer_name=tokenizer_name,
        )

        if para_tok_count <= max_tokens:
            chunk_spans.append(
                (paragraph.start_char, paragraph.end_char)
            )
            continue

        split_spans = _split_oversized_paragraph(
            text=text,
            start_char=paragraph.start_char,
            end_char=paragraph.end_char,
            max_tokens=max_tokens,
            tokenizer_name=tokenizer_name,
        )

        chunk_spans.extend(split_spans)

    if not chunk_spans:
        return []

    # Put skipped spaces and blank lines back into the bounded spans.
    chunk_spans = _make_bounded_spans_contiguous(
        text=text,
        spans=chunk_spans,
        max_tokens=max_tokens,
        tokenizer_name=tokenizer_name,
    )

    chunks: list[TextChunk] = []

    for start_char, end_char in chunk_spans:
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                chunk_text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )

    _validate_chunk_offsets(text=text, chunks=chunks)
    _validate_bounded_chunks_are_contiguous(text=text, chunks=chunks)
    _validate_full_text_reconstruction(text=text, chunks=chunks)

    _validate_bounded_token_limit(
        chunks=chunks,
        max_tokens=max_tokens,
        tokenizer_name=tokenizer_name,
    )

    return chunks


def _split_oversized_paragraph(
    text: str,
    start_char: int,
    end_char: int,
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[int, int]]:
    """Split a large paragraph at sentence boundaries first."""
    sentence_spans = _find_sentence_spans(
        text=text,
        start_char=start_char,
        end_char=end_char,
    )

    split_result: list[tuple[int, int]] = []
    curr_start: int | None = None
    curr_end: int | None = None

    for sentence_start, sentence_end in sentence_spans:
        sentence_text = text[sentence_start:sentence_end]
        sentence_tok_count = count_tokens(
            sentence_text,
            tokenizer_name=tokenizer_name,
        )

        # Fall back to words if one sentence is already too large.
        if sentence_tok_count > max_tokens:
            if curr_start is not None and curr_end is not None:
                split_result.append((curr_start, curr_end))
                curr_start = None
                curr_end = None

            word_spans = _split_span_by_words(
                text=text,
                start_char=sentence_start,
                end_char=sentence_end,
                max_tokens=max_tokens,
                tokenizer_name=tokenizer_name,
            )

            split_result.extend(word_spans)
            continue

        if curr_start is None:
            curr_start = sentence_start
            curr_end = sentence_end
            continue

        candidate_text = text[curr_start:sentence_end]
        candidate_tok_count = count_tokens(
            candidate_text,
            tokenizer_name=tokenizer_name,
        )

        if candidate_tok_count <= max_tokens:
            curr_end = sentence_end

        else:
            split_result.append((curr_start, curr_end))
            curr_start = sentence_start
            curr_end = sentence_end

    if curr_start is not None and curr_end is not None:
        split_result.append((curr_start, curr_end))

    return split_result


def _find_sentence_spans(
    text: str,
    start_char: int,
    end_char: int,
) -> list[tuple[int, int]]:
    """Find sentence spans using periods, question marks, and exclamations."""
    sentence_spans: list[tuple[int, int]] = []
    sentence_start = start_char
    char_idx = start_char

    while char_idx < end_char:
        char = text[char_idx]

        if char in ".!?":
            sentence_end = char_idx + 1

            # Keep repeated punctuation in the same sentence.
            while (
                sentence_end < end_char
                and text[sentence_end] in ".!?"
            ):
                sentence_end += 1

            if sentence_end == end_char or text[sentence_end].isspace():
                clean_start = sentence_start

                while (
                    clean_start < sentence_end
                    and text[clean_start].isspace()
                ):
                    clean_start += 1

                if clean_start < sentence_end:
                    sentence_spans.append(
                        (clean_start, sentence_end)
                    )

                sentence_start = sentence_end
                char_idx = sentence_end
                continue

        char_idx += 1

    clean_start = sentence_start

    while clean_start < end_char and text[clean_start].isspace():
        clean_start += 1

    if clean_start < end_char:
        sentence_spans.append((clean_start, end_char))

    return sentence_spans


def _split_span_by_words(
    text: str,
    start_char: int,
    end_char: int,
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[int, int]]:
    """Split a text span into word groups that fit the token limit."""
    span_text = text[start_char:end_char]
    word_matches = list(re.finditer(r"\S+", span_text))

    split_result: list[tuple[int, int]] = []
    curr_start: int | None = None
    curr_end: int | None = None

    for word_match in word_matches:
        word_start = start_char + word_match.start()
        word_end = start_char + word_match.end()
        word_text = text[word_start:word_end]

        word_tok_count = count_tokens(
            word_text,
            tokenizer_name=tokenizer_name,
        )

        # Fall back to characters when one word is too large.
        if word_tok_count > max_tokens:
            if curr_start is not None and curr_end is not None:
                split_result.append((curr_start, curr_end))
                curr_start = None
                curr_end = None

            char_spans = _split_span_by_characters(
                text=text,
                start_char=word_start,
                end_char=word_end,
                max_tokens=max_tokens,
                tokenizer_name=tokenizer_name,
            )

            split_result.extend(char_spans)
            continue

        if curr_start is None:
            curr_start = word_start
            curr_end = word_end
            continue

        candidate_text = text[curr_start:word_end]
        candidate_tok_count = count_tokens(
            candidate_text,
            tokenizer_name=tokenizer_name,
        )

        if candidate_tok_count <= max_tokens:
            curr_end = word_end

        else:
            split_result.append((curr_start, curr_end))
            curr_start = word_start
            curr_end = word_end

    if curr_start is not None and curr_end is not None:
        split_result.append((curr_start, curr_end))

    return split_result


def _split_span_by_characters(
    text: str,
    start_char: int,
    end_char: int,
    max_tokens: int,
    tokenizer_name: str,
) -> list[tuple[int, int]]:
    """Split a long span at character boundaries that fit the token limit."""
    split_result: list[tuple[int, int]] = []
    curr_start = start_char

    while curr_start < end_char:
        best_end: int | None = None
        candidate_end = curr_start + 1

        while candidate_end <= end_char:
            candidate_text = text[curr_start:candidate_end]
            candidate_tok_count = count_tokens(
                candidate_text,
                tokenizer_name=tokenizer_name,
            )

            if candidate_tok_count > max_tokens:
                break

            best_end = candidate_end
            candidate_end += 1

        if best_end is None:
            raise ValidationError(
                "Maximum token limit is too small for the selected tokenizer."
            )

        split_result.append((curr_start, best_end))
        curr_start = best_end

    return split_result


def chunk_by_fixed_words(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    validate_chunk_settings(
        chunk_size=chunk_size,
        overlap=overlap,
    )

    word_spans: list[tuple[int, int]] = []

    for word_match in re.finditer(r"\S+", text):
        word_spans.append(
            (word_match.start(), word_match.end())
        )

    if not word_spans:
        return []

    chunks: list[TextChunk] = []
    word_step = chunk_size - overlap
    start_word_idx = 0

    while start_word_idx < len(word_spans):
        end_word_idx = min(
            start_word_idx + chunk_size,
            len(word_spans),
        )

        start_char = word_spans[start_word_idx][0]
        end_char = word_spans[end_word_idx - 1][1]

        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                chunk_text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )

        if end_word_idx == len(word_spans):
            break

        start_word_idx += word_step

    _validate_chunk_offsets(text=text, chunks=chunks)

    return chunks


def chunk_text(
    text: str,
    method: str,
    chunk_size: int = 200,
    overlap: int = 40,
    max_tokens: int = 512,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    merge_markdown_headings: bool = False,
) -> list[TextChunk]:
    norm_method = method.lower().strip()

    if norm_method == "paragraph":
        return chunk_by_paragraph(
            text=text,
            merge_markdown_headings=merge_markdown_headings,
        )

    if norm_method == "bounded-paragraph":
        return chunk_by_bounded_paragraph(
            text=text,
            max_tokens=max_tokens,
            tokenizer_name=tokenizer_name,
            merge_markdown_headings=merge_markdown_headings,
        )

    if norm_method == "fixed":
        return chunk_by_fixed_words(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    raise ValidationError(
        "Method must be 'paragraph', 'bounded-paragraph', or 'fixed'."
    )


def _line_content_end(line_start: int, line: str) -> int:
    """Return the position where a line's visible content ends."""
    if line.endswith("\r\n"):
        return line_start + len(line) - 2

    if line.endswith(("\n", "\r")):
        return line_start + len(line) - 1

    return line_start + len(line)