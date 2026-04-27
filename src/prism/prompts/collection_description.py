from __future__ import annotations

SYSTEM = """\
You are an expert text analyst. Your task is to generate features that characterize \
a text collection as a whole.

Each feature must be defined by:
- hypothesis: a declarative statement suitable for NLI scoring \
(e.g. "This text uses technical vocabulary.")

Requirements:
- Features must describe properties present in a substantial portion of the collection
- The hypothesis must be self-contained
- Aim for diverse, non-redundant features

Respond with JSON only:
{"features": [{"hypothesis": "..."}, ...]}
"""

_USER_TEMPLATE = """\
Here are sample texts from the collection:
---
{texts_block}

Generate exactly {n} features that characterize this collection.{language_instruction}
"""


def build_user_message(texts: list[str], n: int, language: str | None = None) -> str:
    texts_block = "\n---\n".join(texts) if texts else "(none)"
    language_instruction = (
        f"\nGenerate the hypothesis for each feature in {language}." if language else ""
    )
    return _USER_TEMPLATE.format(
        texts_block=texts_block,
        n=n,
        language_instruction=language_instruction,
    )
