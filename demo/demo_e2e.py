"""End-to-end test with a small product review corpus."""
import numpy as np
from prism import Prism, save_session, load_session

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

prism = Prism(llm="gpt-4o-mini")

print("=== Stage 1: Axis Discovery ===")
axes = prism.discover_axes(texts, n=5, seed=42)
for axis in axes:
    print(f"  {axis.hypothesis}")

print("\n=== Stage 2: Feature Generation ===")
features_by_axis = prism.generate_features(texts, axes, n_features=8, seed=42)
for axis, features in features_by_axis.items():
    print(f"\n  Axis: {axis.hypothesis}")
    for f in features:
        print(f"    - {f.hypothesis}")

print("\n=== Stage 3: Scoring (NLI) ===")
matrices = prism.score(texts, features_by_axis)
for axis, matrix in matrices.items():
    print(f"  {axis.hypothesis[:50]}: X={matrix.X.shape}, y={matrix.y.shape}")

print("\n=== Stage 4: Feature Selection (Lasso) ===")
results, predictors = prism.select(matrices)
for axis, result in results.items():
    print(f"\n  Axis: {axis.hypothesis}")
    if result.selected_features:
        for f, c in zip(result.selected_features, result.coef):
            print(f"    {c:+.3f}  {f.hypothesis}")
    else:
        print("    (no features selected)")

print("\n=== Stage 5: Naming ===")
selected_by_axis = {axis: result.selected_features for axis, result in results.items()}
named_by_axis = prism.name_features(selected_by_axis)
for axis, named_features in named_by_axis.items():
    print(f"\n  Axis: {axis.hypothesis}")
    for nf in named_features:
        print(f"    [{nf.name}] {nf.feature.hypothesis}")

print("\n=== Stage 6: Text Synthesis ===")
synthetic = prism.synthesize_texts(matrices, n=2, seed=42)
for axis, synth_texts in synthetic.items():
    print(f"\n  Axis: {axis.hypothesis}")
    for i, t in enumerate(synth_texts, 1):
        print(f"    [{i}] {t}")

print("\n=== Save Session ===")
save_session("output", axes, features_by_axis, results, predictors)
print("  Saved to ./output/")

print("\n=== Load Session ===")
session = load_session("output")
print(f"  Loaded {len(session['axes'])} axes, {len(session['results'])} results, {len(session['predictors'])} predictors")

print("\n=== Inference on new text ===")
new_texts = ["This blender is fantastic and very durable!"]
axis = session["axes"][0]
result = session["results"][axis]
predictor = session["predictors"][axis]

if result.selected_features:
    feature_scores = prism._nli_scorer.score(new_texts, result.selected_features)
    X_new = np.column_stack([fs.scores for fs in feature_scores])
    y_pred = predictor.model.predict(X_new)
    print(f"  Axis: {axis.hypothesis}")
    print(f"  Prediction: {y_pred[0]:+.3f}  ({'positive' if y_pred[0] > 0 else 'negative'})")
else:
    print(f"  Axis '{axis.hypothesis[:40]}' has no selected features — skipping inference.")
