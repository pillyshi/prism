SYSTEM = """\
You are an expert text analyst. Your task is to discover interpretable binary axes \
that partition a text collection into meaningful groups.

Each axis must be defined by:
- name: a concise label (2-5 words)
- question: a yes/no question applied to a single text (e.g. "Does this text express a positive emotional tone?")
- hypothesis: a declarative statement suitable for NLI scoring (e.g. "This text has a positive emotional tone.")

Requirements:
- Each axis must be a genuine binary partition (not a continuous spectrum)
- Axes must cover diverse semantic dimensions — maximize variety
- The question and hypothesis must be self-contained (no reference to "the collection" or "other texts")
- Include both obvious and non-obvious axes

Respond with JSON only:
{"axes": [{"name": "...", "question": "...", "hypothesis": "..."}, ...]}
"""

USER_TEMPLATE = """\
Here are sample texts from the collection:

{texts_block}

Generate exactly {n} binary axes for this text collection.
"""


def build_user_message(sampled_texts: list[str], n: int) -> str:
    texts_block = "\n---\n".join(sampled_texts)
    return USER_TEMPLATE.format(texts_block=texts_block, n=n)
