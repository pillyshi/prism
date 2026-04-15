SYSTEM = """\
You are an expert text analyst. Your task is to generate discriminative features \
that explain why texts fall on the positive or negative side of a binary axis.

Each feature must be defined by:
- name: a concise label (3-8 words)
- question: a yes/no question applied to a single text (e.g. "Does this text mention a specific product feature by name?")
- hypothesis: a declarative statement suitable for NLI scoring (e.g. "This text mentions a specific product feature by name.")

Requirements:
- Features must be discriminative: more common on one side than the other
- Include features that favor the positive side AND features that favor the negative side
- The question and hypothesis must be self-contained
- Aim for diverse, non-redundant features

Respond with JSON only:
{"features": [{"name": "...", "question": "...", "hypothesis": "..."}, ...]}
"""

USER_TEMPLATE = """\
Axis: {axis_name}
Positive side (texts where "{axis_question}" is YES):
---
{positive_block}

Negative side (texts where "{axis_question}" is NO):
---
{negative_block}

Generate exactly {n} discriminative features that distinguish the positive side from the negative side.
"""


def build_user_message(
    axis_name: str,
    axis_question: str,
    positive_texts: list[str],
    negative_texts: list[str],
    n: int,
) -> str:
    positive_block = "\n---\n".join(positive_texts) if positive_texts else "(none)"
    negative_block = "\n---\n".join(negative_texts) if negative_texts else "(none)"
    return USER_TEMPLATE.format(
        axis_name=axis_name,
        axis_question=axis_question,
        positive_block=positive_block,
        negative_block=negative_block,
        n=n,
    )
