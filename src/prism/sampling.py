from __future__ import annotations

import random
from typing import Callable


def sample_texts_within_budget(
    texts: list[str],
    token_budget: int,
    tokenizer_fn: Callable[[str], int],
    rng: random.Random | None = None,
) -> list[str]:
    """Return a random subset of texts that fits within the token budget.

    Texts are shuffled, then added one by one until adding the next text
    would exceed token_budget. The order of the returned list follows the
    shuffled order.

    Args:
        texts: Source texts to sample from.
        token_budget: Maximum total tokens allowed.
        tokenizer_fn: Function that returns the token count for a single text.
        rng: Optional Random instance for reproducibility.

    Returns:
        A list of texts whose total token count does not exceed token_budget.
    """
    if rng is None:
        rng = random.Random()

    indices = list(range(len(texts)))
    rng.shuffle(indices)

    selected: list[str] = []
    total_tokens = 0

    for idx in indices:
        text = texts[idx]
        count = tokenizer_fn(text)
        if total_tokens + count > token_budget:
            break
        selected.append(text)
        total_tokens += count

    return selected
