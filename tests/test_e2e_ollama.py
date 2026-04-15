"""End-to-end test using Ollama (local LLM)."""
from langchain_ollama import ChatOllama

from prism import Prism

texts = [
    "This blender is amazing! It crushes ice perfectly and the motor is super powerful.",
    "Terrible product. Broke after two weeks. The plastic feels cheap.",
    "Decent blender for the price. Nothing fancy but gets the job done.",
    "I have had this for 3 years and it still works great. Very durable.",
    "Stopped working after one month. Customer service was unhelpful.",
    "Great value! Makes perfect smoothies every morning.",
    "The blades are sharp and it blends everything smoothly.",
    "Disappointed. The motor burned out after heavy use.",
    "Exactly what I needed. Simple to use and easy to clean.",
    "Horrible experience. Leaked from day one. Returning immediately.",
]

prism = Prism(llm=ChatOllama(model="llama3.2:3b", format="json"))

print("=== Stage 1: Axis Discovery ===")
axes = prism.discover_axes(texts, n=3, seed=42)
for axis in axes:
    print(f"  [{axis.name}] {axis.question}")

print("\n=== Stage 2: Feature Generation ===")
features_by_axis = prism.generate_features(texts, axes, n_features=5, seed=42)
for axis, features in features_by_axis.items():
    print(f"\n  Axis: {axis.name}")
    for f in features:
        print(f"    - {f.name}")

print("\n=== Stage 3: Scoring (NLI) ===")
matrices = prism.score(texts, features_by_axis, method="nli")
for axis, matrix in matrices.items():
    print(f"  {axis.name}: X={matrix.X.shape}, y={matrix.y.shape}")

print("\n=== Stage 4: Feature Selection (Lasso) ===")
results, predictors = prism.select(matrices)
for axis, result in results.items():
    print(f"\n  Axis: {axis.name}")
    if result.selected_features:
        for f, c in zip(result.selected_features, result.coef):
            print(f"    {c:+.3f}  {f.name}")
    else:
        print("    (no features selected)")
