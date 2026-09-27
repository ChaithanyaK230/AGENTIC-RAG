from pathlib import Path

from docling.document_converter import DocumentConverter


def extract_pdf(pdf_path: Path, output_path: Path) -> None:
    """Convert a PDF into Markdown using Docling."""
    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    markdown = result.document.export_to_markdown()

    output_path.write_text(markdown, encoding="utf-8")


def extract_text(file_path: Path, output_path: Path) -> None:
    """Read Markdown/text files as UTF-8 and save as Markdown."""
    text = file_path.read_text(encoding="utf-8")

    output_path.write_text(text, encoding="utf-8")


def extract_file(input_path: Path, output_dir: Path) -> Path:
    """Extract one source file into Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.md"

    if input_path.suffix.lower() == ".pdf":
        extract_pdf(input_path, output_path)

    elif input_path.suffix.lower() in {".md", ".txt"}:
        extract_text(input_path, output_path)

    else:
        raise ValueError(
            f"Unsupported file type: {input_path.suffix}"
        )

    return output_path


def extract_all(input_dir: Path, output_dir: Path) -> None:
    """Extract all supported documents from the input directory."""

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
    )

    for file_path in files:
        print(f"Extracting: {file_path.name}")

        output_path = extract_file(file_path, output_dir)

        print(f"       → {output_path}")


if __name__ == "__main__":
    input_dir = Path("RawData/Data")
    output_dir = Path("ProcessedData/extracted")

    extract_all(input_dir, output_dir)

    print("\nExtraction complete.")