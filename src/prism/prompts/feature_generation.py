SYSTEM = """\
You are an expert text analyst. Your task is to generate discriminative features \
that explain why texts fall on the positive or negative side of a binary axis.

Each feature must be defined by:
- hypothesis: a declarative statement suitable for NLI scoring (e.g. "This text mentions a specific product feature by name.")

Requirements:
- Features must be discriminative: more common on one side than the other
- Include features that favor the positive side AND features that favor the negative side
- The hypothesis must be self-contained
- Aim for diverse, non-redundant features

Respond with JSON only:
{"features": [{"hypothesis": "..."}, ...]}
"""

USER_TEMPLATE = """\
Axis hypothesis: {axis_hypothesis}
Positive side (texts where the axis hypothesis is TRUE):
---
{positive_block}

Negative side (texts where the axis hypothesis is FALSE):
---
{negative_block}

Generate exactly {n} discriminative features that distinguish the positive side from the negative side.{language_instruction}
"""


def build_user_message(
    axis_hypothesis: str,
    positive_texts: list[str],
    negative_texts: list[str],
    n: int,
    language: str | None = None,
) -> str:
    positive_block = "\n---\n".join(positive_texts) if positive_texts else "(none)"
    negative_block = "\n---\n".join(negative_texts) if negative_texts else "(none)"
    language_instruction = f"\nGenerate the hypothesis for each feature in {language}." if language else ""
    return USER_TEMPLATE.format(
        axis_hypothesis=axis_hypothesis,
        positive_block=positive_block,
        negative_block=negative_block,
        n=n,
        language_instruction=language_instruction,
    )
