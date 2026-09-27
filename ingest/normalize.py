from pathlib import Path
import unicodedata

from ftfy import fix_text


INPUT_DIR = Path("ProcessedData/extracted")
OUTPUT_DIR = Path("ProcessedData/normalized")


def normalize_text(text: str) -> str:
    """
    Step 3:
    1. Fix mojibake using ftfy.
    2. Normalize Unicode using NFKC.
    3. Remove zero-width and control characters.
    """

    # Fix encoding problems such as:
    # â€™  -> ’
    # â€“  -> –
    # Ã©   -> é
    text = fix_text(text)

    # Normalize Unicode representations.
    text = unicodedata.normalize("NFKC", text)

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        # Remove control characters.
        if category.startswith("C"):
            # Keep normal whitespace characters.
            if char in "\n\t":
                cleaned.append(char)

            continue

        cleaned.append(char)

    return "".join(cleaned)


def normalize_file(input_path: Path, output_path: Path) -> None:
    """Normalize one Markdown file."""

    text = input_path.read_text(encoding="utf-8")

    normalized = normalize_text(text)

    output_path.write_text(normalized, encoding="utf-8")


def normalize_all(input_dir: Path, output_dir: Path) -> None:
    """Normalize all extracted Markdown files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.md"))

    print(f"Found {len(files)} extracted documents.\n")

    for input_path in files:
        output_path = output_dir / input_path.name

        print(f"Normalizing: {input_path.name}")

        normalize_file(input_path, output_path)

        print(f"        → {output_path}")


if __name__ == "__main__":
    normalize_all(INPUT_DIR, OUTPUT_DIR)

    print("\nNormalization complete.")