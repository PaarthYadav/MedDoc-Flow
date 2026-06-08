"""Unit tests for app/pdf_utils.py."""

import io
import pytest
from app.pdf_utils import extract_text_from_txt


class TestExtractTextFromTxt:
    """Tests for the extract_text_from_txt helper."""

    def test_basic_text_extraction(self):
        content = b"Hello, this is a test medical document."
        file_obj = io.BytesIO(content)
        result = extract_text_from_txt(file_obj)
        assert result == "Hello, this is a test medical document."

    def test_multiline_text(self):
        content = b"Line 1\nLine 2\nLine 3"
        file_obj = io.BytesIO(content)
        result = extract_text_from_txt(file_obj)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_empty_file_raises(self):
        file_obj = io.BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            extract_text_from_txt(file_obj)

    def test_whitespace_only_raises(self):
        file_obj = io.BytesIO(b"   \n\t  ")
        with pytest.raises(ValueError, match="empty"):
            extract_text_from_txt(file_obj)

    def test_non_utf8_bytes_decoded_with_replacement(self):
        # Non-UTF-8 byte sequence should not raise; replaced characters expected.
        content = b"Normal text \xff\xfe"
        file_obj = io.BytesIO(content)
        result = extract_text_from_txt(file_obj)
        assert "Normal text" in result

    def test_string_io_input(self):
        file_obj = io.StringIO("Patient: John Doe\nDiagnosis: Flu")
        result = extract_text_from_txt(file_obj)
        assert "Patient: John Doe" in result
        assert "Diagnosis: Flu" in result
