"""Chat utilities for MedDoc Flow.

This module provides functions for interacting with the Euri AI chat model.
"""

from euriai.langchain import create_chat_model
from typing import Any


def get_chat_model(api_key: str) -> Any:
    """Initialize and return a chat model.

    Args:
        api_key: Euri AI API key.

    Returns:
        Any: Initialized chat model instance.

    Raises:
        ValueError: If the API key is missing or clearly a placeholder.
    """
    normalized_key = api_key.strip() if isinstance(api_key, str) else ""
    if not normalized_key or normalized_key == "your_euri_api_key_here":
        raise ValueError(
            "A valid Euri AI API key is required. "
            "Set the EURI_API_KEY environment variable or update app/config.py."
        )

    chat_model = create_chat_model(
        model_name="gpt-4.1-nano",
        api_key=normalized_key,
        temperature=0.7
    )

    return chat_model


def ask_chat_model(chat_model: Any, prompt: str) -> str:
    """Send a prompt to the chat model and get a response.

    Args:
        chat_model: Initialized chat model instance.
        prompt: The prompt to send to the model.

    Returns:
        str: The model's response.

    Raises:
        RuntimeError: If the model returns an unexpected response type.
    """
    try:
        response = chat_model.invoke(prompt)

        # Extract content from the response
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as exc:
        raise RuntimeError(f"Chat model request failed: {exc}") from exc
