
import re
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
import ollama


CAPTION_PATTERN = re.compile(
    r"(?im)^\s*((?:figure|fig\.|table|algorithm|listing|code)"
    r"\s*[\dIVX.-]*\s*[:.\-]?\s*.*)$"
)


def extract_captions(page_text):
    """Find figure, table, algorithm, and code captions."""
    return [
        match.group(1).strip()
        for match in CAPTION_PATTERN.finditer(page_text or "")
    ]


def extract_pdf_visual_content(
    pdf_path,
    output_dir="data/processed_pages",
    vision_model=None,
    ocr_language="eng",
):
    """
    Extract searchable visual information from a PDF.

    Returns records containing text, source, page, content type,
    captions, and optional vision descriptions.
    """
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)

    image_dir = Path(output_dir) / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for page_index, page in enumerate(document):
        page_number = page_index + 1
        page_text = page.get_text("text") or ""
        captions = extract_captions(page_text)

        # Render pages that may contain visual information.
        embedded_images = page.get_images(full=True)
        needs_visual_processing = (
            bool(embedded_images)
            or bool(captions)
            or len(page_text.strip()) < 100
        )

        if not needs_visual_processing:
            continue

        image_path = image_dir / f"page_{page_number:04d}.png"

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        )
        pixmap.save(str(image_path))

        # OCR helps recover text embedded in images and scanned pages.
        try:
            with Image.open(image_path) as image:
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang=ocr_language,
                ).strip()
        except Exception as exc:
            ocr_text = ""
            print(
                f"OCR failed for {pdf_path.name}, "
                f"page {page_number}: {exc}"
            )

        # Avoid duplicating text already extracted from the PDF.
        normalized_page_text = re.sub(
            r"\s+", " ", page_text
        ).strip().lower()
        normalized_ocr_text = re.sub(
            r"\s+", " ", ocr_text
        ).strip().lower()

        if normalized_ocr_text and normalized_ocr_text not in normalized_page_text:
            combined_text = (
                page_text.strip()
                + "\n\nOCR text:\n"
                + ocr_text
            ).strip()
        else:
            combined_text = page_text.strip() or ocr_text

        visual_description = ""

        # Optional: interpret the rendered page using a vision model.
        if vision_model:
            try:
                response = ollama.chat(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Inspect this research-paper page. "
                                "Describe any diagram, flowchart, graph, "
                                "table, equation, or code listing. "
                                "For diagrams, identify visible components "
                                "and arrows or data flow. For graphs, report "
                                "only readable labels and values. For code, "
                                "transcribe only readable code and mark "
                                "uncertain portions. Do not invent details. "
                                "Include the figure or table number if visible."
                            ),
                            "images": [str(image_path)],
                        }
                    ],
                    options={"temperature": 0},
                )
                visual_description = (
                    response["message"]["content"].strip()
                )
            except Exception as exc:
                print(
                    f"Vision analysis failed for {pdf_path.name}, "
                    f"page {page_number}: {exc}"
                )

        # Save one searchable record per processed page.
        if combined_text or captions or visual_description:
            content_parts = []

            if captions:
                content_parts.append(
                    "Captions found:\n" + "\n".join(captions)
                )

            if combined_text:
                content_parts.append(
                    "Extracted page text:\n" + combined_text
                )

            if visual_description:
                content_parts.append(
                    "Vision description:\n" + visual_description
                )

            records.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "content_type": "visual_page",
                    "captions": captions,
                    "text": "\n\n".join(content_parts),
                    "image_path": str(image_path),
                }
            )

    document.close()
    return records
