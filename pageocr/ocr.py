import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import re

# -------------------------------
# CONFIG — paths from environment
# On Linux/Docker: tesseract is at /usr/bin/tesseract (installed via apt)
# On Windows local: set TESSERACT_PATH in .env
# Poppler is on PATH in Linux so no explicit path needed
# -------------------------------
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "/usr/bin/tesseract")
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)  # None = use system PATH (Linux)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# -------------------------------
# HEADING DETECTION
# -------------------------------
def detect_heading(line: str) -> str:
    """Convert text into markdown headers based on heuristics."""
    line = line.strip()

    # ALL CAPS → likely section
    if line.isupper() and len(line.split()) < 10:
        return f"## {line}"

    # Numbered headings → sub-sections
    if re.match(r"^\d+(\.\d+)*\s+", line):
        return f"### {line}"

    # Long sentences → normal text
    return line


# -------------------------------
# OCR PDF → TEXT
# -------------------------------
def pdf_to_text(pdf_path: str) -> str:
    print("Converting PDF to images...")

    # Pass poppler_path only if explicitly set (Windows local dev)
    # On Linux/Docker poppler is on PATH so None is correct
    kwargs = {}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    images = convert_from_path(pdf_path, **kwargs)

    all_text = []
    for i, img in enumerate(images):
        print(f"OCR on page {i + 1}/{len(images)}...")
        text = pytesseract.image_to_string(img)
        all_text.append(text)

    return "\n".join(all_text)


# -------------------------------
# TEXT → MARKDOWN
# -------------------------------
def text_to_markdown(raw_text: str) -> str:
    lines = raw_text.split("\n")
    md_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            md_lines.append("")
            continue
        md_line = detect_heading(line)
        md_lines.append(md_line)
    return "\n".join(md_lines)


# -------------------------------
# SAVE MARKDOWN
# -------------------------------
def save_markdown(md_text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Markdown saved at: {output_path}")


# -------------------------------
# MAIN (local dev only)
# -------------------------------
if __name__ == "__main__":
    pdf_path = os.environ.get("TEST_PDF_PATH", "test.pdf")
    output_md = pdf_path.replace(".pdf", ".md")

    raw_text = pdf_to_text(pdf_path)
    print("Converting to markdown...")
    md_text = text_to_markdown(raw_text)
    save_markdown(md_text, output_md)