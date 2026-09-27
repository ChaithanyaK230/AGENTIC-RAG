from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Common cleaning
# ---------------------------------------------------------------------------

def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive newlines into 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def strip_trailing_whitespace(text: str) -> str:
    """Remove unnecessary spaces at the end of lines."""
    return "\n".join(line.rstrip() for line in text.splitlines())


# ---------------------------------------------------------------------------
# MDX / LangChain documentation
# ---------------------------------------------------------------------------

def remove_documentation_index(text: str) -> str:
    """
    Remove the documentation index block.

    Example:

    ## Documentation Index
    ...
    ## Actual Content

    becomes:

    ## Actual Content
    """

    pattern = re.compile(
        r"(?ms)^##\s+Documentation Index\b.*?(?=^#{1,6}\s+\S|\Z)"
    )

    return pattern.sub("", text)


def clean_images(text: str) -> str:
    """
    Convert HTML image tags into useful Markdown text.

    <img src="..." alt="Architecture diagram">

    becomes:

    [Image: Architecture diagram]
    """

    def replace_image(match: re.Match) -> str:
        tag = match.group(0)

        alt_match = re.search(
            r'alt\s*=\s*["\']([^"\']*)["\']',
            tag,
            flags=re.IGNORECASE,
        )

        if alt_match and alt_match.group(1).strip():
            return f"[Image: {alt_match.group(1).strip()}]"

        return ""

    return re.sub(
        r"<img\b[^>]*>",
        replace_image,
        text,
        flags=re.IGNORECASE,
    )


def unwrap_mdx_blocks(text: str) -> str:
    """
    Convert MDX components into normal Markdown.

    <Note>
    Important information.
    </Note>

    becomes:

    > Important information.
    """

    components = [
        "Note",
        "Tip",
        "Warning",
    ]

    for component in components:
        pattern = re.compile(
            rf"(?ms)<{component}\b[^>]*>\s*(.*?)\s*</{component}>"
        )

        def replace(match: re.Match) -> str:
            content = match.group(1).strip()

            if not content:
                return ""

            lines = content.splitlines()

            quoted = "\n".join(
                f"> {line}" if line.strip() else ">"
                for line in lines
            )

            return f"\n{quoted}\n"

        text = pattern.sub(replace, text)

    return text


def remove_fence_theme_metadata(text: str) -> str:
    """
    Remove theme={{...}} from Markdown code-fence info strings.

    Example:

    ```typescript theme={{...}}

    becomes:

    ```typescript
    """

    return re.sub(
        r"(```[^\n]*?)\s+theme=\{\{.*?\}\}",
        r"\1",
        text,
        flags=re.DOTALL,
    )


def clean_code_groups(text: str) -> str:
    """
    Simplify <CodeGroup> blocks.

    We keep the first code example inside each CodeGroup and remove
    repeated provider variants.

    This is intentionally conservative: it does not modify ordinary
    code fences outside CodeGroup.
    """

    pattern = re.compile(
        r"(?ms)<CodeGroup\b[^>]*>(.*?)</CodeGroup>"
    )

    def replace(match: re.Match) -> str:
        content = match.group(1).strip()

        # Find fenced code blocks.
        blocks = re.findall(
            r"```.*?```",
            content,
            flags=re.DOTALL,
        )

        if blocks:
            # Keep the first complete code example.
            return "\n\n" + blocks[0] + "\n\n"

        # If no fenced block exists, preserve the textual content.
        return "\n\n" + content + "\n\n"

    return pattern.sub(replace, text)


def unwrap_remaining_jsx(text: str) -> str:
    """
    Remove remaining JSX/HTML-style tags while preserving their content.

    Example:

    <Tabs>
    useful content
    </Tabs>

    becomes:

    useful content
    """

    return re.sub(
        r"</?[A-Za-z][^>]*>",
        "",
        text,
    )


def clean_mdx(text: str) -> str:
    """Clean LangChain/Mintlify Markdown/MDX documentation."""

    text = remove_documentation_index(text)
    text = clean_images(text)
    text = unwrap_mdx_blocks(text)
    text = clean_code_groups(text)
    text = remove_fence_theme_metadata(text)
    text = unwrap_remaining_jsx(text)

    return text


# ---------------------------------------------------------------------------
# Web / Markdown text
# ---------------------------------------------------------------------------

def remove_markdown_urls(text: str) -> str:
    """
    Convert:

    [Vector Database](https://example.com)

    into:

    Vector Database
    """

    return re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r"\1",
        text,
    )


def clean_web_text(text: str) -> str:
    """Clean web-sourced Markdown/text."""

    text = remove_markdown_urls(text)

    return text


# ---------------------------------------------------------------------------
# Academic papers
# ---------------------------------------------------------------------------

def dehyphenate_line_breaks(text: str) -> str:
    """
    Fix words broken across lines.

    retriev-
    al

    becomes:

    retrieval
    """

    return re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text,
    )


def remove_arxiv_stamps(text: str) -> str:
    """
    Remove common arXiv identifiers/stamps.
    """

    patterns = [
        r"arXiv:\d{4}\.\d+(?:v\d+)?",
        r"arXiv preprint arXiv:\S+",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    return text


def remove_page_numbers(text: str) -> str:
    """
    Remove lines that contain only a page number.

    Example:

    12

    """

    return re.sub(
        r"(?m)^\s*\d{1,4}\s*$\n?",
        "",
        text,
    )


def remove_references(text: str) -> str:
    """
    Remove the References/Bibliography section until the end.

    This assumes References is a top-level Markdown heading.
    """

    pattern = re.compile(
        r"(?ims)^#{1,2}\s+(?:references|bibliography)\s*$.*\Z"
    )

    return pattern.sub("", text)


def clean_paper(text: str) -> str:
    """Clean extracted academic paper Markdown."""

    text = dehyphenate_line_breaks(text)
    text = remove_arxiv_stamps(text)
    text = remove_page_numbers(text)
    text = remove_references(text)

    return text


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

def classify_document(path: Path) -> str:
    """
    Determine which source-specific cleaner to use.

    This project has three source categories:
        - docs
        - web
        - paper
    """

    name = path.name.lower()

    if name in {"agents.md", "mcp.md"}:
        return "docs"

    if "vector database" in name:
        return "web"

    if path.suffix.lower() == ".pdf":
        return "paper"

    return "web"


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_document(text: str, source_type: str) -> str:
    """Apply the appropriate source-specific cleaner."""

    if source_type == "docs":
        text = clean_mdx(text)

    elif source_type == "web":
        text = clean_web_text(text)

    elif source_type == "paper":
        text = clean_paper(text)

    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Common cleanup applied to every source.
    text = collapse_blank_lines(text)
    text = strip_trailing_whitespace(text)

    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def clean_all(input_dir: Path, output_dir: Path) -> None:
    """Clean all normalized Markdown documents."""

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.md"))

    print(f"Found {len(files)} normalized documents.\n")

    for input_path in files:
        source_type = classify_document(input_path)

        text = input_path.read_text(encoding="utf-8")

        before = len(text)

        cleaned = clean_document(
            text,
            source_type,
        )

        after = len(cleaned)

        removed = (
            ((before - after) / before) * 100
            if before
            else 0
        )

        output_path = output_dir / input_path.name

        output_path.write_text(
            cleaned,
            encoding="utf-8",
        )

        print(
            f"{input_path.name}"
            f" | type={source_type}"
            f" | before={before:,}"
            f" | after={after:,}"
            f" | removed={removed:.1f}%"
        )


if __name__ == "__main__":
    input_dir = Path("ProcessedData/normalized")
    output_dir = Path("ProcessedData/cleaned")

    clean_all(input_dir, output_dir)

    print("\nCleaning complete.")