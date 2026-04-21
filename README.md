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

Axes are discovered automatically by Prism, or can be provided manually. Each axis is defined by a single hypothesis:

```json
{
  "hypothesis": "This text has a positive emotional tone."
}
```

Example axes discovered from a product review corpus:

| Hypothesis | Positive side | Negative side |
|---|---|---|
| `"This text has a positive emotional tone."` | expresses satisfaction or delight | expresses frustration or disappointment |
| `"This text mentions concrete product details."` | specific features, measurements | vague or general impressions |
| `"This text indicates intention to repurchase."` | plans to buy again | no intention to repurchase |

### Feature
A fine-grained property that explains *why* a text falls on one side of an axis.

For the axis `"This text mentions concrete product details."`, Prism might generate features such as:

- `"This text mentions a specific product feature by name."`
- `"This text includes a numeric rating or measurement."`
- `"This text compares the product to a competitor or previous version."`

Features share the same structure as axes — the framework is recursive.

### Feature Vector
Each text is scored against all features using NLI (continuous entailment scores):

| Text | mentions specific feature | includes numeric rating | compares to competitor | ... |
|------|--------------------------|------------------------|----------------------|-----|
| doc1 | 1.0 | 0.9 | 0.0 | ... |
| doc2 | 0.0 | 0.1 | 0.8 | ... |

### Named Feature
After feature selection, human-readable names can be assigned to finalized features as a display-layer step:

```python
named = prism.name_features(selected_by_axis)
# named[axis][0].name  → "mentions specific feature"
# named[axis][0].feature.hypothesis  → "This text mentions a specific product feature by name."
```

---

## Pipeline

```
Stage 1: Axis Discovery
  Text collection
    → LLM generates axes (hypothesis only)
    → Texts are labeled positive/negative per axis (NLI)

Stage 2: Feature Generation
  Per axis: positive/negative examples
    → LLM generates discriminative features (hypothesis only)
    → Texts are scored per feature (NLI → continuous)

Stage 3: Feature Selection
  Feature vectors + labels
    → Lasso / Elastic Net selects predictive features
    → Output: sparse, interpretable feature set

Stage 4 (optional): Naming
  Finalized hypotheses
    → LLM assigns short human-readable labels
```

Axis discovery and feature generation share the same structure — the pipeline is recursive. An axis is a coarse feature; a feature is a fine-grained axis.

---

## Quickstart

```python
from prism import Prism

prism = Prism(llm="gpt-4o", nli_model="cross-encoder/nli-deberta-v3-large")

# Stage 1: Discover axes from a text collection
axes = prism.discover_axes(texts, n=20)

# Stage 2: Generate features per axis
features = prism.generate_features(texts, axes)

# Stage 3: Score texts → feature matrices (NLI)
matrices = prism.score(texts, features)

# Stage 4: Select predictive features
results, predictors = prism.select(matrices)

# Stage 5 (optional): Assign human-readable names to selected features
selected_by_axis = {axis: result.selected_features for axis, result in results.items()}
named_by_axis = prism.name_features(selected_by_axis)
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

### User-provided labels

If you already have axes of interest and labeled data, NLI-based auto-labeling can be skipped entirely by passing `axes_labels` to both `generate_features()` and `score()`:

```python
from prism import Prism, Axis, AxisLabels

my_axis = Axis(hypothesis="This text expresses positive sentiment.")
# Classification mode: +1.0 / -1.0. Regression mode: continuous [0, 1].
my_labels = AxisLabels(axis=my_axis, labels=[1.0, -1.0, 1.0, ...])

prism = Prism(llm="gpt-4o")

features_by_axis = prism.generate_features(texts, [my_axis], axes_labels=[my_labels])
matrices = prism.score(texts, features_by_axis, axes_labels=[my_labels])
results, predictors = prism.select(matrices)
```

---

### Multi-run axis merging

Running Prism multiple times with different random seeds improves axis coverage. Use `merge_axes()` to consolidate axes from multiple runs using NLI entailment — axes where either hypothesis implies the other are merged into a single representative:

```python
# Run axis discovery multiple times
axes_run1 = prism.discover_axes(texts, n=20, seed=0)
axes_run2 = prism.discover_axes(texts, n=20, seed=1)
axes_run3 = prism.discover_axes(texts, n=20, seed=2)

# Merge into a unified, non-redundant axis set (NLI-based)
merged_axes = prism.merge_axes([axes_run1, axes_run2, axes_run3])

# Continue the pipeline with the merged axes
features = prism.generate_features(texts, merged_axes)
```

---

## Design Principles

- **Interpretable by construction** — every feature is grounded in a natural language hypothesis
- **Recursive** — the same pipeline applies at the axis level and the feature level
- **NLI-native** — hypotheses are scored directly by NLI models; no discrete yes/no conversion needed
- **Model-agnostic** — any LLM for generation, any NLI model for scoring

---

## Roadmap

- [x] Axis discovery (LLM-based)
- [x] Feature generation (contrastive, positive/negative examples)
- [x] NLI-based scoring
- [x] Lasso / Elastic Net selection
- [x] Multi-run axis merging (NLI entailment-based)
- [x] Post-fixation naming (display layer)
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
