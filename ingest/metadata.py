from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re

import yaml


INPUT_DIR = Path("ProcessedData/cleaned")
OUTPUT_DIR = Path("ProcessedData/metadata")


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_doc_id(filename: str) -> str:
    """Create a stable document ID from the filename."""

    name = Path(filename).stem.lower()

    # Replace spaces and special characters with "-"
    name = re.sub(r"[^a-z0-9]+", "-", name)

    return name.strip("-")


def extract_title(text: str, fallback: str) -> str:
    """
    Extract the first Markdown H1 as the document title.

    Example:
        # Retrieval-Augmented Generation

    returns:
        Retrieval-Augmented Generation
    """

    match = re.search(
        r"(?m)^#\s+(.+?)\s*$",
        text,
    )

    if match:
        return match.group(1).strip()

    return fallback


def determine_source_type(path: Path) -> str:
    """Determine the source category."""

    name = path.name.lower()

    if name in {"agents.md", "mcp.md"}:
        return "docs"

    if "vector database" in name:
        return "web"

    if path.suffix.lower() == ".md":
        return "paper"

    return "unknown"


def build_metadata(
    path: Path,
    text: str,
) -> dict:
    """Build metadata for one document."""

    return {
        "doc_id": create_doc_id(path.name),
        "title": extract_title(
            text,
            fallback=path.stem,
        ),
        "source_path": str(
            Path("RawData/Data") / path.name
        ),
        "source_type": determine_source_type(path),
        "sha256": calculate_sha256(path),
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def add_front_matter(
    metadata: dict,
    text: str,
) -> str:
    """Add YAML front matter to Markdown."""

    yaml_metadata = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
    ).strip()

    return (
        "---\n"
        f"{yaml_metadata}\n"
        "---\n\n"
        f"{text.strip()}\n"
    )


def process_file(
    input_path: Path,
    output_path: Path,
) -> None:
    """Add metadata to one cleaned document."""

    text = input_path.read_text(
        encoding="utf-8"
    )

    metadata = build_metadata(
        input_path,
        text,
    )

    output_text = add_front_matter(
        metadata,
        text,
    )

    output_path.write_text(
        output_text,
        encoding="utf-8",
    )


def process_all(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Process all cleaned Markdown documents."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        input_dir.glob("*.md")
    )

    print(
        f"Found {len(files)} cleaned documents.\n"
    )

    for input_path in files:

        output_path = (
            output_dir / input_path.name
        )

        process_file(
            input_path,
            output_path,
        )

        print(
            f"{input_path.name}"
            f" → {output_path}"
        )


if __name__ == "__main__":
    process_all(
        INPUT_DIR,
        OUTPUT_DIR,
    )

    print(
        "\nMetadata enrichment complete."
    )