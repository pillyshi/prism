"""End-to-end demo with a small product review corpus."""
from pathlib import Path

import numpy as np

from prism import (
    FeatureSelector,
    Prism,
    TextSynthesizer,
    evaluate_fit,
    evaluate_generation,
)
from prism.llm import LLMClient

texts = [
    "This blender is amazing! It crushes ice perfectly and the motor is super powerful. Will definitely buy again.",
    "Terrible product. Broke after two weeks. The plastic feels cheap and the lid doesn't seal properly.",
    "Decent blender for the price. Nothing fancy but gets the job done for smoothies.",
    "I've had this for 3 years and it still works great. Very durable and easy to clean.",
    "Stopped working after one month. Customer service was unhelpful. Complete waste of money.",
    "Great value! Makes perfect smoothies every morning. A bit loud but I can live with that.",
    "The blades are sharp and it blends everything smoothly. My old blender was nowhere near this good.",
    "Disappointed. The motor burned out after heavy use. Not suitable for daily use.",
    "Exactly what I needed. Simple to use, easy to clean, and does the job well.",
    "Horrible experience. Leaked from day one. Returning immediately.",
    "Solid build quality. Handles frozen fruit without any issues. Happy with this purchase.",
    "Not worth the money. Very noisy and the buttons are hard to press.",
    "Love it! Makes the creamiest smoothies. The variable speed settings are very useful.",
    "Fell apart within a week. The handle cracked and the seal failed. Very poor quality.",
    "Good blender overall. A little pricey but the performance justifies the cost.",
    "Works as advertised. Nothing outstanding but reliable for everyday use.",
    "The best blender I've ever owned. Powerful, quiet, and incredibly easy to clean.",
    "Broke on first use. The blade assembly came loose. Very dangerous design flaw.",
    "Average product. Gets the job done but I expected better quality for this price.",
    "Outstanding performance. Blends everything smoothly and the noise level is surprisingly low.",
]

# Binary sentiment labels (1 = positive, 0 = negative)
y = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)

llm = LLMClient("gpt-4o-mini")
prism = Prism(llm=llm, nli_model="cross-encoder/nli-deberta-v3-large")

# ---------------------------------------------------------------------------
print("=== Stage 1: Feature Generation ===")
# ---------------------------------------------------------------------------
features = prism.generate_features(texts, n=15, seed=42)
for f in features:
    print(f"  - {f.hypothesis}")

# ---------------------------------------------------------------------------
print("\n=== Stage 2: Scoring (NLI) ===")
# ---------------------------------------------------------------------------
X = prism.score(texts, features)
print(f"  X shape: {X.shape}")
print("  Feature mean scores:")
for f, mean_score in zip(features, X.mean(axis=0)):
    print(f"    {mean_score:.3f}  {f.hypothesis}")

# ---------------------------------------------------------------------------
print("\n=== Stage 3: Feature Naming ===")
# ---------------------------------------------------------------------------
named = prism.name_features(features)
for nf in named:
    print(f"  [{nf.name}]  {nf.feature.hypothesis}")

# ---------------------------------------------------------------------------
print("\n=== Stage 4: Feature Dependency Analysis ===")
# ---------------------------------------------------------------------------
selector = FeatureSelector(r2_threshold=0.9).fit(X, features)
print("  R²  hypothesis")
for dep in selector.dependencies_:
    flag = " ← redundant" if dep.r2 > 0.9 else ""
    print(f"  {dep.r2:.2f}  {dep.feature.hypothesis}{flag}")

X2, features2 = selector.transform(X, features)
print(f"\n  After transform: {X.shape[1]} → {X2.shape[1]} features")

# ---------------------------------------------------------------------------
print("\n=== Stage 5: Fit (y ~ X) ===")
# ---------------------------------------------------------------------------
result = prism.fit(X2, y, features2)
print(f"  Scoring: {result.scoring}, score={result.score:.3f}")
print(f"  intercept: {result.intercept:+.3f}")
pairs = sorted(zip(result.coef, result.features), key=lambda x: abs(x[0]), reverse=True)
for coef, f in pairs:
    print(f"    {coef:+.3f}  {f.hypothesis}")

# ---------------------------------------------------------------------------
print("\n=== Stage 6: Text Synthesis ===")
# ---------------------------------------------------------------------------
Path("output").mkdir(exist_ok=True)

synthesizer = TextSynthesizer().fit(X2, features2)
synthesizer.save("output/synthesizer.json")
print("  Saved to output/synthesizer.json")

loaded = TextSynthesizer.load("output/synthesizer.json")
X_sampled = X2[:3]
synth_texts = loaded.synthesize(X_sampled, llm=llm, n_levels=3)
for i, t in enumerate(synth_texts, 1):
    print(f"  [{i}] {t}")

# ---------------------------------------------------------------------------
print("\n=== Stage 7: Synthesis Evaluation ===")
# ---------------------------------------------------------------------------
# evaluate_fit: how well the Gaussian captures the original distribution
fit_eval = evaluate_fit(X2, X_sampled)
print(f"  Fit Wasserstein (mean): {fit_eval.wasserstein.mean():.3f}")

# evaluate_generation: how faithfully the LLM followed the target feature vectors
X_scored = prism.score(synth_texts, features2)
gen_eval = evaluate_generation(X_sampled, X_scored)
print(f"  Generation MAE (mean): {gen_eval.mae.mean():.3f}")
print("  Per-feature MAE:")
for f, mae in zip(features2, gen_eval.mae):
    print(f"    {mae:.3f}  {f.hypothesis}")
