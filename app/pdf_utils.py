"""PDF utilities for MedDoc Flow.

This module provides functions for extracting text from PDF and plain-text
documents.
"""

from pypdf import PdfReader
from typing import BinaryIO, Union
import io


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_file: Binary file object containing the PDF data.

    Returns:
        str: Extracted text from all pages of the PDF.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    try:
        pdf_reader = PdfReader(pdf_file)
        extracted_pages = []

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

        text = "".join(extracted_pages)

        if not text.strip():
            raise ValueError("No extractable text found in the PDF.")

        return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read PDF: {exc}") from exc


def extract_text_from_txt(txt_file: Union[BinaryIO, io.IOBase]) -> str:
    """Extract text from a plain-text file.

    Args:
        txt_file: Binary or text file object.

    Returns:
        str: Decoded text content of the file.

    Raises:
        ValueError: If the file cannot be decoded or is empty.
    """
    try:
        raw = txt_file.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = raw

        if not text.strip():
            raise ValueError("The text file is empty.")

        return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read text file: {exc}") from exc
