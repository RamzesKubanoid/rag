"""Day 1 — document loading and basic cleaning.

Reads plain-text files from the knowledge_base/ folder, normalizes their
formatting and returns them as a list of `Document` objects, ready for
chunking on Day 2.
"""

import logging
import re
import unicodedata
from pathlib import Path

from schemas import Document

logger = logging.getLogger(__name__)

# Formats treated as plain text. Parsing PDF/DOCX is out of scope for
# this educational project.
SUPPORTED_EXTENSIONS = {".txt", ".md"}

_MULTI_SPACES = re.compile(r" {2,}")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Normalize whitespace and formatting of a raw document string.

    Steps:
      1. Unicode NFKC normalization (unifies ligatures, exotic spaces,
         full-width characters, etc.).
      2. Windows / old-Mac line endings are converted to Unix newlines.
      3. Tabs and non-breaking spaces become regular spaces.
      4. Runs of spaces collapse into a single space.
      5. Whitespace is stripped from the edges of every line.
      6. Several blank lines in a row collapse into a single blank line.

    A single blank line is intentionally preserved: it marks a paragraph
    boundary, and the Day 2 chunker will split on paragraphs first.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ").replace("\u00a0", " ")
    text = _MULTI_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _extract_title(text: str, fallback: str) -> str:
    """Use the first non-empty line as the title, else the file name."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line if len(line) <= 80 else line[:77] + "..."
    return fallback


def load_document(path: Path, doc_id: str) -> Document | None:
    """Load and clean a single file; returns None if it ends up empty."""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "%s is not valid UTF-8 — decoding with errors='replace'", path.name
        )
        raw = path.read_text(encoding="utf-8", errors="replace")

    text = clean_text(raw)
    if not text:
        logger.warning("Skipping %s: empty after cleaning", path.name)
        return None

    document = Document(
        doc_id=doc_id,
        source=path.name,
        path=str(path),
        title=_extract_title(text, fallback=path.stem),
        text=text,
    )
    logger.debug(
        "%s: %d raw chars -> %d cleaned chars",
        path.name,
        len(raw),
        document.num_chars,
    )
    return document


def load_knowledge_base(folder: str | Path = "knowledge_base") -> list[Document]:
    """Load every supported file from `folder` as a Document.

    Files are processed in alphabetical order so doc_ids stay stable
    between runs.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {folder.resolve()} — "
            "create it and put at least 5 .txt files inside."
        )

    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No {sorted(SUPPORTED_EXTENSIONS)} files found in {folder.resolve()}"
        )

    logger.info("Found %d candidate files in %s/", len(files), folder)

    documents: list[Document] = []
    for index, path in enumerate(files, start=1):
        document = load_document(path, doc_id=f"doc_{index:03d}")
        if document is not None:
            documents.append(document)

    logger.info(
        "Loaded %d documents, %d characters in total",
        len(documents),
        sum(d.num_chars for d in documents),
    )
    return documents
