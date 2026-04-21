SYSTEM = """\
You are a creative writer. Generate a single text that satisfies the given conditions.
- Naturally reflect the TRUE conditions and avoid the FALSE conditions.
- Output only the generated text itself, with no explanation.
- Match the style and domain implied by the axis.
"""

USER_TEMPLATE = """\
Axis: {axis_hypothesis}
Generate a text that satisfies the following conditions:

{conditions_block}{language_instruction}
"""


def build_user_message(
    axis_hypothesis: str,
    conditions: list[tuple[str, float, bool]],
    language: str | None = None,
) -> str:
    lines = [
        f'- "{hypothesis}" → {score:.2f} ({"TRUE" if is_true else "FALSE"})'
        for hypothesis, score, is_true in conditions
    ]
    conditions_block = "\n".join(lines)
    language_instruction = f"\n\nRespond in {language}." if language else ""
    return USER_TEMPLATE.format(
        axis_hypothesis=axis_hypothesis,
        conditions_block=conditions_block,
        language_instruction=language_instruction,
    )
