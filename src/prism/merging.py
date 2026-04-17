from __future__ import annotations

from .llm import BaseLLMClient
from .models import Axis
from .prompts import axis_merging as prompts


class AxisMerger:
    """Merges axes from multiple discovery runs using an LLM."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    def merge(self, axes_per_run: list[list[Axis]]) -> list[Axis]:
        """Consolidate axes from multiple runs into a unified, non-redundant set.

        Args:
            axes_per_run: One list of Axis objects per run.

        Returns:
            Merged list of Axis objects.
        """
        axes_dicts = [
            [{"name": a.name, "question": a.question, "hypothesis": a.hypothesis} for a in run]
            for run in axes_per_run
        ]
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.build_user_message(axes_dicts)},
        ]
        result = self._llm.complete_json(messages)
        return [
            Axis(name=item["name"], question=item["question"], hypothesis=item["hypothesis"])
            for item in result.get("merged_axes", [])
        ]
