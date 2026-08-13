"""Render RESUME_OVERVIEW.md into a formatted Word document.

Deliberately a narrow renderer rather than a general Markdown-to-docx converter:
it handles exactly the constructs the overview uses (headings, bullets, tables,
block quotes, fenced code, bold/inline-code runs) and would rather raise on
something unexpected than silently drop content from a document someone is about
to send to an employer.

Usage:
    .venv/Scripts/python.exe scripts/build_overview_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

SOURCE = Path("RESUME_OVERVIEW.md")
TARGET = Path("AutoML Architect - Project Overview.docx")

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x59, 0x59, 0x59)
CODE_COLOR = RGBColor(0xA3, 0x1D, 0x1D)

# `code`, **bold**, or *italic* — captured so runs can be styled individually.
INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*))")


def add_inline(paragraph, text: str) -> None:
    """Append text to a paragraph, styling inline code / bold / italic runs."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = CODE_COLOR
        elif piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)


def style_document(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 20), (2, 15), (3, 12)):
        heading = document.styles[f"Heading {level}"]
        heading.font.name = "Calibri"
        heading.font.size = Pt(size)
        heading.font.color.rgb = ACCENT
        heading.font.bold = True
        heading.paragraph_format.space_before = Pt(14 if level < 3 else 10)
        heading.paragraph_format.space_after = Pt(4)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Read a pipe table starting at `start`; return its rows and the next index."""

    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    rows = [cells(lines[start])]
    index = start + 1
    if index < len(lines) and set(lines[index].replace("|", "").strip()) <= set("-: "):
        index += 1  # the alignment row carries no content
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        rows.append(cells(lines[index]))
        index += 1
    return rows, index


def render_table(document: Document, rows: list[list[str]]) -> None:
    width = max(len(r) for r in rows)
    table = document.add_table(rows=0, cols=width)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    for position, row in enumerate(rows):
        cells = table.add_row().cells
        for column in range(width):
            text = row[column] if column < len(row) else ""
            paragraph = cells[column].paragraphs[0]
            add_inline(paragraph, text)
            if position == 0:
                for run in paragraph.runs:
                    run.bold = True
    document.add_paragraph()


def render(markdown: str, document: Document) -> None:
    lines = markdown.splitlines()
    index = 0
    skipped_title = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.space_after = Pt(10)
            run = paragraph.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
            continue

        if stripped.startswith("|"):
            rows, index = parse_table(lines, index)
            render_table(document, rows)
            continue

        if stripped.startswith("---"):
            index += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1 and not skipped_title:
                skipped_title = True  # the title is set separately, above
                index += 1
                continue
            document.add_heading(text, level=min(level, 3))
            index += 1
            continue

        if stripped.startswith(">"):
            body = stripped.lstrip(">").strip()
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.3)
            paragraph.paragraph_format.space_after = Pt(8)
            if body:
                add_inline(paragraph, body)
                for run in paragraph.runs:
                    run.italic = True
                    if run.font.color.rgb is None:
                        run.font.color.rgb = MUTED
            index += 1
            continue

        if stripped.startswith(("- ", "* ")):
            indent = len(line) - len(line.lstrip())
            style = "List Bullet 2" if indent >= 2 else "List Bullet"
            paragraph = document.add_paragraph(style=style)
            add_inline(paragraph, stripped[2:].strip())
            index += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            paragraph = document.add_paragraph(style="List Number")
            add_inline(paragraph, re.sub(r"^\d+\.\s", "", stripped))
            index += 1
            continue

        # Join wrapped prose into one paragraph so Word reflows it naturally.
        body = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("#", "-", "*", ">", "|", "```")) or re.match(
                r"^\d+\.\s", nxt
            ):
                break
            body.append(nxt)
            index += 1
        paragraph = document.add_paragraph()
        add_inline(paragraph, " ".join(body))


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}")
        return 1

    document = Document()
    style_document(document)

    title = document.add_heading("AutoML Architect", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle = document.add_paragraph()
    run = subtitle.add_run(
        "An autonomous multi-agent AI data scientist — project overview for "
        "resumes, portfolios, and interviews"
    )
    run.italic = True
    run.font.size = Pt(11)
    run.font.color.rgb = MUTED

    render(SOURCE.read_text(encoding="utf-8"), document)
    document.save(TARGET)

    size_kb = TARGET.stat().st_size / 1024
    print(f"wrote {TARGET}  ({size_kb:.1f} KB, {len(document.paragraphs)} paragraphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
