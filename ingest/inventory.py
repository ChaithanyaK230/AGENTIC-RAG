from pathlib import Path
import hashlib
import re


RAW_DIR = Path("RawData/Data")


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 fingerprint of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def make_doc_id(filename: str) -> str:
    """Convert filename into a stable document ID."""
    name = Path(filename).stem.lower()

    # Replace spaces and special characters with "-"
    name = re.sub(r"[^a-z0-9]+", "-", name)

    # Remove leading/trailing "-"
    return name.strip("-")


def detect_type(path: Path) -> str:
    """Detect basic document type."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return "pdf"

    if suffix in {".md", ".mdx"}:
        return "markdown"

    if suffix == ".txt":
        return "text/markdown"

    return "unknown"


def inventory_documents():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {RAW_DIR}")

    files = sorted(
        path for path in RAW_DIR.iterdir()
        if path.is_file()
    )

    print(f"Found {len(files)} documents\n")

    documents = []

    for path in files:
        document = {
            "filename": path.name,
            "doc_id": make_doc_id(path.name),
            "type": detect_type(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

        documents.append(document)

        print(f"File:       {document['filename']}")
        print(f"Document ID:{document['doc_id']}")
        print(f"Type:       {document['type']}")
        print(f"Size:       {document['size_bytes']:,} bytes")
        print(f"SHA-256:    {document['sha256']}")
        print("-" * 60)

    return documents


if __name__ == "__main__":
    inventory_documents()