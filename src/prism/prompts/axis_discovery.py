SYSTEM = """\
You are an expert text analyst. Your task is to discover interpretable binary axes \
that partition a text collection into meaningful groups.

Each axis must be defined by:
- hypothesis: a declarative statement suitable for NLI scoring (e.g. "This text has a positive emotional tone.")

Requirements:
- Each axis must be a genuine binary partition (not a continuous spectrum)
- Axes must cover diverse semantic dimensions — maximize variety
- The hypothesis must be self-contained (no reference to "the collection" or "other texts")
- Include both obvious and non-obvious axes

Respond with JSON only:
{"axes": [{"hypothesis": "..."}, ...]}
"""

USER_TEMPLATE = """\
Here are sample texts from the collection:

{texts_block}

Generate exactly {n} binary axes for this text collection.{language_instruction}
"""


def build_user_message(sampled_texts: list[str], n: int, language: str | None = None) -> str:
    texts_block = "\n---\n".join(sampled_texts)
    language_instruction = f"\nGenerate the hypothesis for each axis in {language}." if language else ""
    return USER_TEMPLATE.format(texts_block=texts_block, n=n, language_instruction=language_instruction)
