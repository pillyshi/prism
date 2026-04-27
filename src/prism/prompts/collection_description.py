from __future__ import annotations

SYSTEM = """\
You are an expert text analyst. Your task is to generate features that describe \
properties of individual texts in a collection.

Each feature must be defined by:
- hypothesis: a declarative statement about a single text, suitable for NLI scoring \
(e.g. "This text expresses satisfaction with the product." or \
"This text mentions issues with build quality.")

Requirements:
- Each hypothesis must be a statement about a single text, starting with "This text"
- Features must capture properties that are meaningfully present in some texts \
but not all — avoid trivially universal or trivially absent properties
- The hypothesis must be self-contained
- Aim for diverse, non-redundant features

Respond with JSON only:
{"features": [{"hypothesis": "..."}, ...]}
"""

_USER_TEMPLATE = """\
Here are sample texts from the collection:
---
{texts_block}

Generate exactly {n} features that describe properties of individual texts \
commonly found in this collection.{language_instruction}
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
