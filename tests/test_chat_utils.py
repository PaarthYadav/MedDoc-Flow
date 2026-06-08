"""Unit tests for app/chat_utils.py."""

import sys
import pytest
from unittest.mock import MagicMock, patch

# euriai is not installed in the test environment; mock the entire package so
# that the import in app/chat_utils succeeds without the real dependency.
_mock_euriai = MagicMock()
sys.modules.setdefault("euriai", _mock_euriai)
sys.modules.setdefault("euriai.langchain", _mock_euriai.langchain)

from app.chat_utils import get_chat_model, ask_chat_model  # noqa: E402


class TestGetChatModel:
    """Tests for get_chat_model."""

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="valid Euri AI API key"):
            get_chat_model("")

    def test_placeholder_api_key_raises(self):
        with pytest.raises(ValueError, match="valid Euri AI API key"):
            get_chat_model("your_euri_api_key_here")

    def test_whitespace_api_key_raises(self):
        with pytest.raises(ValueError, match="valid Euri AI API key"):
            get_chat_model("   ")

    def test_valid_key_calls_create_chat_model(self):
        mock_model = MagicMock()
        with patch("app.chat_utils.create_chat_model", return_value=mock_model) as mock_create:
            result = get_chat_model("valid-key-abc123")
            mock_create.assert_called_once_with(
                model_name="gpt-4.1-nano",
                api_key="valid-key-abc123",
                temperature=0.7,
            )
            assert result is mock_model


class TestAskChatModel:
    """Tests for ask_chat_model."""

    def test_returns_content_attribute_when_present(self):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is the answer."
        mock_model.invoke.return_value = mock_response

        result = ask_chat_model(mock_model, "What is the diagnosis?")
        assert result == "This is the answer."

    def test_falls_back_to_str_when_no_content_attr(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = "plain string response"

        result = ask_chat_model(mock_model, "What are the symptoms?")
        assert result == "plain string response"

    def test_raises_runtime_error_on_exception(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ConnectionError("network failure")

        with pytest.raises(RuntimeError, match="Chat model request failed"):
            ask_chat_model(mock_model, "Tell me about the patient.")
