from __future__ import annotations

SYSTEM = """\
You are a creative writer. Generate a single text that satisfies the given conditions.
- Naturally reflect the TRUE or high-scoring conditions.
- Avoid properties marked as FALSE or low-scoring.
- Output only the generated text itself, with no explanation.
"""

_USER_TEMPLATE = """\
Generate a text that satisfies the following conditions:

{conditions_block}{language_instruction}
"""


def build_user_message(
    conditions: list[tuple[str, str]],
    language: str | None = None,
) -> str:
    """Build the user message for collection-based text synthesis.

    Args:
        conditions: List of (hypothesis, formatted_label) pairs.
        language: If specified, instruct the LLM to respond in this language.
    """
    if conditions:
        lines = [f'- "{hypothesis}" → {label}' for hypothesis, label in conditions]
        conditions_block = "\n".join(lines)
    else:
        conditions_block = "(no specific conditions)"
    language_instruction = f"\n\nRespond in {language}." if language else ""
    return _USER_TEMPLATE.format(
        conditions_block=conditions_block,
        language_instruction=language_instruction,
    )
