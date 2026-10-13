import numpy as np
import pandas as pd
import torch
from data import load_split, ROOT
from autoencoder import DeepSoftmaxAutoencoder, train_deep_softmax_ae, EvaluateAdapter
from evaluate import predict, clip_report, song_vote, plot_pca, set_seed

# Ensure results subdirectories exist
for sub in ("figures", "metrics", "models"):
    (ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)

rows = []
for rms_norm in [False, True]:
    for split in ["song", "clip"]:
        tag = f"{split}_norm{int(rms_norm)}"
        print(f"\n================ Running Deep Softmax Autoencoder: {tag} ================")
        data = load_split(split, rms_norm=rms_norm)

        for seed in [0, 1, 2]:
            set_seed(seed)
            raw_model = DeepSoftmaxAutoencoder()
            trained_model, hist = train_deep_softmax_ae(raw_model, data, gamma=0.9, seed=seed)
            model = EvaluateAdapter(trained_model)

            row = {"rms_norm": rms_norm, "split": split, "seed": seed, "epochs_run": len(hist)}

            for name in ["train", "dev", "test"]:
                X, y, sid = data[name]
                _, logits = predict(model, X)
                row[name] = float((logits.argmax(1) == y).mean())

            X, y, sid = data["test"]
            emb, logits = predict(model, X)
            row["test_song_vote"] = song_vote(logits, y, sid) if split == "song" else np.nan
            rows.append(row)
            print(f"Seed {seed}: train={row['train']:.3f}, dev={row['dev']:.3f}, test={row['test']:.3f}, song_vote={row['test_song_vote']}")

            if seed == 0:  # Save figures and weights once per experimental setting
                clip_report(y, logits, f"deep_softmax_{tag}", outdir=ROOT / "results")
                plot_pca(logits, y, f"Deep Softmax AE 4-dim Logits ({tag})",
                         ROOT / f"results/figures/pca_deep_softmax_logits_{tag}.png")
                plot_pca(emb, y, f"Deep Softmax AE 64-dim Bottleneck ({tag})",
                         ROOT / f"results/figures/pca_deep_softmax_hidden_{tag}.png")
                torch.save(trained_model.state_dict(), ROOT / f"results/models/deep_softmax_{tag}.pt")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/metrics/deep_softmax_runs.csv", index=False)

summary = (df.groupby(["rms_norm", "split"])[["train", "dev", "test", "test_song_vote"]]
             .agg(["mean", "std"]).round(3))
summary.to_csv(ROOT / "results/metrics/deep_softmax.csv")

print("\n================ FINAL DEEP SOFTMAX AUTOENCODER SUMMARY ================")
print(summary)
