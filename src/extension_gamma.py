import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_split, ROOT
from autoencoder import DeepSoftmaxAutoencoder, train_deep_softmax_ae, EvaluateAdapter
from evaluate import predict, song_vote

# Ensure output subdirectories exist
(ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

gamma_values = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
seeds = [0, 1, 2]

print("================ Running Extension: Loss Weight (gamma) Sensitivity Analysis ================")
data = load_split("song", rms_norm=False)
X_test, y_test, sid_test = data["test"]

rows = []

for gamma in gamma_values:
    print(f"\n--- Testing gamma = {gamma:.2f} ---")
    for seed in seeds:
        raw_model = DeepSoftmaxAutoencoder()
        # For gamma=1.0 (pure reconstruction), train_deep_softmax_ae handles MSE only
        trained_model, hist = train_deep_softmax_ae(raw_model, data, gamma=gamma, seed=seed)
        model = EvaluateAdapter(trained_model)

        # Predict
        _, logits = predict(model, X_test)
        clip_acc = float((logits.argmax(1) == y_test).mean())
        vote_acc = song_vote(logits, y_test, sid_test)

        row = {
            "gamma": gamma,
            "seed": seed,
            "epochs_run": len(hist),
            "test_clip_acc": clip_acc,
            "test_song_vote_acc": vote_acc
        }
        rows.append(row)
        print(f"Seed {seed}: clip_acc={clip_acc:.3f}, song_vote={vote_acc:.3f}")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/metrics/gamma_ablation_runs.csv", index=False)

summary = (df.groupby("gamma")[["test_clip_acc", "test_song_vote_acc"]]
             .agg(["mean", "std"]).round(3))
summary.to_csv(ROOT / "results/metrics/gamma_ablation.csv")

print("\n================ GAMMA ABLATION SUMMARY ================")
print(summary)

# Plotting Gamma Sensitivity
plt.figure(figsize=(7, 5))
gammas = summary.index.values
clip_means = summary[("test_clip_acc", "mean")].values
clip_stds = summary[("test_clip_acc", "std")].values
vote_means = summary[("test_song_vote_acc", "mean")].values
vote_stds = summary[("test_song_vote_acc", "std")].values

plt.errorbar(gammas, clip_means, yerr=clip_stds, fmt='-o', capsize=4, label="Test Clip Accuracy", color="tab:blue")
plt.errorbar(gammas, vote_means, yerr=vote_stds, fmt='-s', capsize=4, label="Test Song Vote Accuracy", color="tab:orange")
plt.axvline(x=0.90, color="gray", linestyle="--", label="Paper Default (gamma=0.9)")

plt.title("Effect of Reconstruction Loss Weight (γ) on Test Accuracy")
plt.xlabel("Reconstruction Loss Weight γ  (0 = Pure Classification, 1 = Pure AE)")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = ROOT / "results/figures/gamma_ablation_plot.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved ablation plot to {plot_path}")
