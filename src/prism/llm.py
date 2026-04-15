from __future__ import annotations

import json
from typing import Any

import tiktoken
from openai import OpenAI


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API with token counting."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the content string."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        """Send a request expecting JSON output and return the parsed object."""
        content = self.complete(messages, response_format={"type": "json_object"})
        return json.loads(content)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text using the model's tokenizer."""
        return len(self._encoding.encode(text))
