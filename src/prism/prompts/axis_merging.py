SYSTEM = """\
You are an expert text analyst. Your task is to consolidate a set of binary axes \
discovered from multiple analysis runs into a unified, non-redundant axis set.

Each axis is defined by:
- name: a concise label (2-5 words)
- question: a yes/no question applied to a single text (e.g. "Does this text express a positive emotional tone?")
- hypothesis: a declarative statement suitable for NLI scoring (e.g. "This text has a positive emotional tone.")

Requirements:
- Merge axes that are semantically equivalent or nearly identical
- Preserve axes that cover distinct semantic dimensions
- For merged axes, either select the best representative or generate a new definition
- Maximize coverage of the original axes while minimizing redundancy
- Output axes must remain genuine binary partitions
- The question and hypothesis must be self-contained (no reference to "the collection" or "other texts")

Respond with JSON only:
{"merged_axes": [{"name": "...", "question": "...", "hypothesis": "..."}, ...]}
"""

USER_TEMPLATE = """\
The following axes were discovered across {n_runs} analysis runs:

{axes_block}

Consolidate these into a unified, non-redundant set of axes.
"""


def build_user_message(axes_per_run: list[list[dict]]) -> str:
    lines = []
    for i, axes in enumerate(axes_per_run, 1):
        lines.append(f"## Run {i}")
        for ax in axes:
            lines.append(f"- {ax['name']}: {ax['hypothesis']}")
    axes_block = "\n".join(lines)
    return USER_TEMPLATE.format(n_runs=len(axes_per_run), axes_block=axes_block)
