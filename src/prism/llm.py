from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import tiktoken
from openai import OpenAI


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol defining the interface required by all LLM client implementations."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...
    def complete_json(self, messages: list[dict[str, str]]) -> Any: ...
    def count_tokens(self, text: str) -> int: ...


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API with token counting."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request and return the content string."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        """Send a request expecting JSON output and return the parsed object."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return json.loads(content)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text using the model's tokenizer."""
        return len(self._encoding.encode(text))


class LangChainLLMClient:
    """LLM client backed by any LangChain BaseChatModel.

    Supports Ollama, llama.cpp, and any other LangChain-compatible provider.

    Example::

        from langchain_ollama import ChatOllama
        client = LangChainLLMClient(ChatOllama(model="llama3.2", format="json"))
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Invoke the LangChain model and return the response string."""
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = self._model.invoke(lc_messages)
        return response.content

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        """Invoke the model and extract a JSON object from the response."""
        content = self.complete(messages)
        return _extract_json(content)

    def count_tokens(self, text: str) -> int:
        """Return an approximate token count for the given text."""
        try:
            return self._model.get_num_tokens(text)
        except (NotImplementedError, AttributeError):
            # Fallback: ~4 characters per token (reasonable for English)
            return len(text) // 4


def _extract_json(text: str) -> Any:
    """Extract a JSON object from a string, tolerating surrounding prose.

    Tries in order:
    1. Direct json.loads (model output is clean JSON)
    2. Extract from a markdown code block (```json ... ```)
    3. Extract the first {...} or [...] block via regex
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try first {...} or [...] block
    brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from model response:\n{text}")
