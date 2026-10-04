import hashlib
import re
from dataclasses import asdict, dataclass

from document_chunking_utility.chunkers import TextChunk
from document_chunking_utility.io import SourceDocument
from document_chunking_utility.tokenizer import count_tokens


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_file: str
    source_path: str
    method: str
    chunk_index: int
    chunk_text: str
    start_char: int
    end_char: int
    char_count: int
    word_count: int
    chunk_size: int | None = None
    overlap: int | None = None
    token_count: int | None = None
    max_tokens: int | None = None
    tokenizer: str | None = None


def build_chunk_records(
    document: SourceDocument,
    chunks: list[TextChunk],
    method: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | None = None,
) -> list[ChunkRecord]:
    """Build metadata records for all chunks in one document."""
    chunk_records: list[ChunkRecord] = []

    for chunk in chunks:
        chunk_record = build_chunk_record(
            document=document,
            chunk=chunk,
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )

        chunk_records.append(chunk_record)

    return chunk_records


def build_chunk_record(
    document: SourceDocument,
    chunk: TextChunk,
    method: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | None = None,
) -> ChunkRecord:
    """Build metadata for one text chunk."""
    norm_method = method.lower().strip()

    token_count: int | None = None
    saved_max_tokens: int | None = None
    saved_tokenizer: str | None = None

    # Token information is only used for bounded paragraph chunks.
    if norm_method == "bounded-paragraph":
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for bounded paragraph metadata."
            )

        if max_tokens is None:
            raise ValueError(
                "Maximum tokens are required for bounded paragraph metadata."
            )

        saved_tokenizer = tokenizer.strip()
        saved_max_tokens = max_tokens

        token_count = count_tokens(
            text=chunk.chunk_text,
            tokenizer_name=saved_tokenizer,
        )

    return ChunkRecord(
        chunk_id=build_chunk_id(
            source_path=document.source_path,
            chunk_index=chunk.chunk_index,
            method=norm_method,
            chunk_size=chunk_size,
            overlap=overlap,
            max_tokens=saved_max_tokens,
            tokenizer=saved_tokenizer,
        ),
        source_file=document.source_file,
        source_path=document.source_path,
        method=norm_method,
        chunk_index=chunk.chunk_index,
        chunk_text=chunk.chunk_text,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        char_count=len(chunk.chunk_text),
        word_count=len(chunk.chunk_text.split()),
        chunk_size=chunk_size,
        overlap=overlap,
        token_count=token_count,
        max_tokens=saved_max_tokens,
        tokenizer=saved_tokenizer,
    )


def build_chunk_id(
    source_path: str,
    chunk_index: int,
    method: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | None = None,
) -> str:
    """Build a stable ID for one chunk."""
    src_slug = _slugify_source_path(source_path)
    src_hash = _short_source_hash(source_path)

    config_slug = _build_chunk_config_slug(
        method=method,
        chunk_size=chunk_size,
        overlap=overlap,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )

    return f"{src_slug}_{src_hash}_{config_slug}_{chunk_index:03d}"


def chunk_record_to_dict(
    record: ChunkRecord,
) -> dict[str, object]:
    """Convert one chunk record into a dictionary."""
    return asdict(record)


def chunk_records_to_dicts(
    records: list[ChunkRecord],
) -> list[dict[str, object]]:
    """Convert multiple chunk records into dictionaries."""
    record_dicts: list[dict[str, object]] = []

    for record in records:
        record_dicts.append(
            chunk_record_to_dict(record)
        )

    return record_dicts


def _build_chunk_config_slug(
    method: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | None = None,
) -> str:
    """Build the chunk ID section that describes its settings."""
    norm_method = method.lower().strip()
    method_slug = _slugify_text(norm_method)

    # Paragraph chunking does not need any extra settings.
    if norm_method == "paragraph":
        return method_slug

    # Fixed chunking uses the chunk size and overlap.
    if norm_method == "fixed":
        if chunk_size is None:
            raise ValueError(
                "Chunk size is required for fixed chunk IDs."
            )

        if overlap is None:
            raise ValueError(
                "Overlap is required for fixed chunk IDs."
            )

        return f"{method_slug}_cs{chunk_size}_ov{overlap}"

    # Bounded paragraph chunking uses a token limit and tokenizer.
    if norm_method == "bounded-paragraph":
        if max_tokens is None:
            raise ValueError(
                "Maximum tokens are required for bounded paragraph chunk IDs."
            )

        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for bounded paragraph chunk IDs."
            )

        tokenizer_slug = _slugify_text(tokenizer.strip())

        if not tokenizer_slug:
            raise ValueError(
                "Tokenizer is required for bounded paragraph chunk IDs."
            )

        return (
            f"{method_slug}_mt{max_tokens}_tk_{tokenizer_slug}"
        )

    # Unknown methods normally get rejected before reaching this point.
    if method_slug:
        return method_slug

    return "unknown"


def _short_source_hash(source_path: str) -> str:
    """Create a short hash from a normalized source path."""
    norm_path = _normalize_source_path(source_path)
    encoded_path = norm_path.encode("utf-8")
    full_hash = hashlib.sha1(encoded_path).hexdigest()

    return full_hash[:8]


def _slugify_source_path(source_path: str) -> str:
    """Convert a source path into a chunk-ID-friendly value."""
    norm_path = _normalize_source_path(source_path)

    if norm_path == "document":
        return "document"

    path_parts = norm_path.split("/")
    path_parts[-1] = _slugify_filename(path_parts[-1])

    raw_slug = "_".join(path_parts)
    path_slug = _slugify_text(raw_slug)

    return path_slug or "document"


def _normalize_source_path(source_path: str) -> str:
    """Normalize path separators and remove empty path parts."""
    norm_path = str(source_path).replace("\\", "/")
    norm_path = norm_path.strip("/")

    path_parts = []

    for part in norm_path.split("/"):
        if part:
            path_parts.append(part)

    if not path_parts:
        return "document"

    return "/".join(path_parts)


def _slugify_filename(filename: str) -> str:
    """Convert a filename into a chunk-ID-friendly value."""
    if "." not in filename:
        return _slugify_text(filename) or "document"

    stem, extension = filename.rsplit(".", 1)

    stem_slug = _slugify_text(stem)
    extension_slug = _slugify_text(extension)

    if stem_slug and extension_slug:
        return f"{stem_slug}_{extension_slug}"

    if stem_slug:
        return stem_slug

    if extension_slug:
        return f"document_{extension_slug}"

    return "document"


def _slugify_text(text: str) -> str:
    """Convert text into a lowercase underscore-separated slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text)
    slug = slug.strip("_").lower()
    slug = re.sub(r"_+", "_", slug)

    return slug