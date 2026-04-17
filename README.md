# Prism 🔬
### Language-Defined Feature Space for Interpretable Text Analysis

Prism is a framework for discovering interpretable features from text collections using Large Language Models. Instead of relying on black-box embeddings, Prism defines a feature space in natural language — axes and features are human-readable by design.

---

## Motivation

Traditional text representations (embeddings, BoW) are powerful but opaque. LLM prompting is flexible but offers no structured output. Prism bridges these worlds: it uses LLMs to *discover* and *define* a feature space in natural language, then scores each text against that space to produce interpretable feature vectors.

---

## Core Concepts

### Axis
A binary partition of a text collection with a natural language label.

Axes are discovered automatically by Prism, or can be provided manually. Each axis is defined by a triple:

```json
{
  "name": "emotional tone",
  "question": "Does this text express a predominantly positive emotional tone?",
  "hypothesis": "This text has a positive emotional tone."
}
```

Example axes discovered from a product review corpus:

| Axis | Positive side | Negative side |
|------|--------------|---------------|
| `emotional tone` | expresses satisfaction or delight | expresses frustration or disappointment |
| `specificity of feedback` | mentions concrete product details | gives only vague or general impressions |
| `purchase intent` | indicates intention to buy again | indicates no intention to repurchase |

### Feature
A fine-grained property that explains *why* a text falls on one side of an axis.

For the axis `specificity of feedback`, Prism might generate features such as:

- `"mentions a specific product feature by name"`
- `"includes a numeric rating or measurement"`
- `"compares the product to a competitor or previous version"`

Features share the same triple structure as axes — the framework is recursive.

### Feature Vector
Each text is scored against all features using QA (discrete) or NLI (continuous):

| Text | mentions specific feature | includes numeric rating | compares to competitor | ... |
|------|--------------------------|------------------------|----------------------|-----|
| doc1 | 1.0 | 0.9 | 0.0 | ... |
| doc2 | 0.0 | 0.1 | 0.8 | ... |

---

## Pipeline

```
Stage 1: Axis Discovery
  Text collection
    → LLM generates axes (name + question + hypothesis)
    → Texts are labeled positive/negative per axis (QA or NLI)

Stage 2: Feature Generation
  Per axis: positive/negative examples
    → LLM generates discriminative features (name + question + hypothesis)
    → Texts are scored per feature (QA → discrete, NLI → continuous)

Stage 3: Feature Selection
  Feature vectors + labels
    → Lasso / Elastic Net selects predictive features
    → Output: sparse, interpretable feature set
```

Axis discovery and feature generation share the same structure — the pipeline is recursive. An axis is a coarse feature; a feature is a fine-grained axis.

---

## Scoring Methods

| Method | Output | Model |
|--------|--------|-------|
| **QA** | Discrete (Yes / No) | LLM |
| **NLI** | Continuous (entailment score) | NLI model (e.g. DeBERTa-MNLI) |

QA is more flexible; NLI is cheaper and faster at scale.

---

## Quickstart

```python
from prism import Prism

prism = Prism(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")

# Stage 1: Discover axes from a text collection
axes = prism.discover_axes(texts, n=20)


# Stage 2: Generate features per axis
features = prism.generate_features(texts, axes)

# Stage 3: Score texts → feature vectors
vectors = prism.score(texts, features, method="nli")

# Stage 4: Select predictive features
selected = prism.select(vectors, labels, method="lasso")
```

### Local models (Ollama / llama.cpp)

Prism supports any [LangChain](https://python.langchain.com/)-compatible LLM, enabling fully local inference with Ollama or llama.cpp:

```python
from langchain_ollama import ChatOllama
from prism import Prism

prism = Prism(llm=ChatOllama(model="llama3.2", format="json"))
```

```python
from langchain_community.llms import LlamaCpp
from prism import Prism

prism = Prism(llm=LlamaCpp(model_path="./models/llama-3.2.gguf"))
```

Any model that implements LangChain's `BaseChatModel` interface works as a drop-in replacement.

---

### Multi-run axis merging

Running Prism multiple times with different random seeds improves axis coverage. Use `merge_axes()` to consolidate axes from multiple runs before feature generation:

```python
# Run axis discovery multiple times
axes_run1 = prism.discover_axes(texts, n=20, seed=0)
axes_run2 = prism.discover_axes(texts, n=20, seed=1)
axes_run3 = prism.discover_axes(texts, n=20, seed=2)

# Merge into a unified, non-redundant axis set (LLM-based)
merged_axes = prism.merge_axes([axes_run1, axes_run2, axes_run3])

# Continue the pipeline with the merged axes
features = prism.generate_features(texts, merged_axes)
```

---

## Design Principles

- **Interpretable by construction** — every feature has a natural language name
- **Recursive** — the same pipeline applies at the axis level and the feature level
- **Flexible scoring** — QA for classification-like features, NLI for graded features
- **Model-agnostic** — any LLM for generation, any NLI model for scoring

---

## Roadmap

- [x] Axis discovery (LLM-based)
- [x] Feature generation (contrastive, positive/negative examples)
- [ ] QA-based scoring
- [x] NLI-based scoring
- [x] Lasso / Elastic Net selection
- [x] Multi-run axis merging (LLM-based consolidation)
- [ ] Feature deduplication (semantic similarity + correlation)
- [ ] Visualization of feature space
- [ ] Evaluation suite (predictive performance vs. embeddings baseline)

---

## Related Work

- Balek et al. (2025) — [LLM-based feature generation for interpretable ML](https://arxiv.org/abs/2409.07132)
- Yin et al. (2019) — NLI as zero-shot text classifier
- LogiPart (2025) — LLM hypothesis generation + NLI propagation

---

## Citation

```bibtex
@software{prism2026,
  title  = {Prism: Language-Defined Feature Space for Interpretable Text Analysis},
  year   = {2026},
}
```
