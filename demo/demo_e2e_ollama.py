"""End-to-end demo using Ollama (local LLM) for feature generation and synthesis."""
import numpy as np
from langchain_ollama import ChatOllama

from prism import FeatureSelector, Prism, TextSynthesizer
from prism.llm import LLMClient

texts = [
    "This blender is amazing! It crushes ice perfectly and the motor is super powerful.",
    "Terrible product. Broke after two weeks. The plastic feels cheap.",
    "Decent blender for the price. Nothing fancy but gets the job done.",
    "I've had this for 3 years and it still works great. Very durable.",
    "Stopped working after one month. Customer service was unhelpful.",
    "Great value! Makes perfect smoothies every morning.",
    "The blades are sharp and it blends everything smoothly.",
    "Disappointed. The motor burned out after heavy use.",
    "Exactly what I needed. Simple to use and easy to clean.",
    "Horrible experience. Leaked from day one. Returning immediately.",
]

y = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0], dtype=float)

# Ollama for LLM tasks (feature generation, naming, synthesis)
ollama = ChatOllama(model="llama3.2:3b", format="json")
# Separate client for text synthesis (can be any LLM)
synth_llm = ChatOllama(model="llama3.2:3b")

prism = Prism(llm=ollama, nli_model="cross-encoder/nli-deberta-v3-large")

print("=== Stage 1: Feature Generation ===")
features = prism.generate_features(texts, n=8, seed=42)
for f in features:
    print(f"  - {f.hypothesis}")

print("\n=== Stage 2: Scoring (NLI) ===")
X = prism.score(texts, features)
print(f"  X shape: {X.shape}")

print("\n=== Stage 3: Feature Naming ===")
named = prism.name_features(features)
for nf in named:
    print(f"  [{nf.name}]  {nf.feature.hypothesis}")

print("\n=== Stage 4: Feature Dependency Analysis ===")
selector = FeatureSelector(r2_threshold=0.9).fit(X, features)
for dep in selector.dependencies_:
    flag = " ← redundant" if dep.r2 > 0.9 else ""
    print(f"  R²={dep.r2:.2f}  {dep.feature.hypothesis}{flag}")

X2, features2 = selector.transform(X, features)
print(f"\n  After transform: {X.shape[1]} → {X2.shape[1]} features")

print("\n=== Stage 5: Fit (y ~ X) ===")
result = prism.fit(X2, y, features2)
print(f"  Scoring: {result.scoring}, score={result.score:.3f}")
pairs = sorted(zip(result.coef, result.features), key=lambda x: abs(x[0]), reverse=True)
for coef, f in pairs:
    print(f"    {coef:+.3f}  {f.hypothesis}")

print("\n=== Stage 6: Text Synthesis ===")
lengths = np.array([len(t) for t in texts])
synthesizer = TextSynthesizer().fit(features2, lengths=lengths)
synth_texts = synthesizer.synthesize(X2[:2], llm=synth_llm, n_levels=2)
for i, t in enumerate(synth_texts, 1):
    print(f"  [{i}] {t}")
