SYSTEM = """\
You are an expert text analyst. Given a list of declarative hypotheses, \
assign each a concise human-readable label (3-8 words).

Respond with JSON only:
{"names": ["...", "..."]}

The names list must be the same length as the input hypotheses list, in the same order.
"""

USER_TEMPLATE = """\
Assign a short label to each of the following hypotheses:{language_instruction}

{hypotheses_block}
"""


def build_user_message(hypotheses: list[str], language: str | None = None) -> str:
    hypotheses_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(hypotheses))
    language_instruction = f"\nGenerate labels in {language}." if language else ""
    return USER_TEMPLATE.format(
        language_instruction=language_instruction,
        hypotheses_block=hypotheses_block,
    )
