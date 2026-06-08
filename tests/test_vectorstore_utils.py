"""Unit tests for app/vectorstore_utils.py."""

import sys
import pytest
from unittest.mock import MagicMock

# langchain_community is not installed in the test environment; provide stubs
# so that the module-level imports in vectorstore_utils succeed.
_mock_lc_community = MagicMock()
sys.modules.setdefault("langchain_community", _mock_lc_community)
sys.modules.setdefault("langchain_community.vectorstores", _mock_lc_community.vectorstores)
sys.modules.setdefault("langchain_community.embeddings", _mock_lc_community.embeddings)

from app.vectorstore_utils import retrieve_relevant_docs, retrive_relavent_docs  # noqa: E402


class TestRetrieveRelevantDocs:
    """Tests for retrieve_relevant_docs."""

    def test_returns_docs_from_similarity_search(self):
        mock_store = MagicMock()
        mock_docs = [MagicMock(), MagicMock(), MagicMock()]
        mock_store.similarity_search.return_value = mock_docs

        result = retrieve_relevant_docs(mock_store, "What is the diagnosis?")

        mock_store.similarity_search.assert_called_once_with(
            "What is the diagnosis?", k=3
        )
        assert result == mock_docs

    def test_custom_k_value(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        retrieve_relevant_docs(mock_store, "query", k=5)

        mock_store.similarity_search.assert_called_once_with("query", k=5)


class TestBackwardCompatAlias:
    """Ensure deprecated alias still works correctly."""

    def test_alias_delegates_to_retrieve_relevant_docs(self):
        mock_store = MagicMock()
        mock_docs = [MagicMock()]
        mock_store.similarity_search.return_value = mock_docs

        result = retrive_relavent_docs(mock_store, "old function name", k=2)

        mock_store.similarity_search.assert_called_once_with(
            "old function name", k=2
        )
        assert result == mock_docs
