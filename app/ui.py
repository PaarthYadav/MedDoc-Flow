"""UI components for MedDoc Flow.

This module provides Streamlit UI components for the application.
"""

import streamlit as st
from typing import List, Optional


def pdf_uploader() -> Optional[List]:
    """Display a file uploader for PDF and plain-text documents.

    Returns:
        Optional[List]: List of uploaded files, or None if no files uploaded.
    """
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more medical documents (PDF or plain-text)"
    )

    return uploaded_files if uploaded_files else None
