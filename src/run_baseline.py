import numpy as np, pandas as pd, torch
from data import load_split, ROOT
from baseline import TwoLayerNN, train_classifier
from evaluate import predict, clip_report, song_vote, plot_pca

for sub in ("figures", "metrics", "models"):
    (ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)

rows = []
for split in ["song", "clip"]:
    data = load_split(split)
    for seed in [0, 1, 2]:
        model, hist = train_classifier(TwoLayerNN(), data, seed=seed)
        row = {"split": split, "seed": seed, "epochs_run": len(hist)}
        for name in ["train", "dev", "test"]:
            X, y, sid = data[name]
            _, logits = predict(model, X)
            row[name] = float((logits.argmax(1) == y).mean())
        X, y, sid = data["test"]
        emb, logits = predict(model, X)
        row["test_song_vote"] = song_vote(logits, y, sid) if split == "song" else np.nan
        rows.append(row)
        print(row)

        if seed == 0:   # save figures and weights once per split
            clip_report(y, logits, f"baseline_{split}", outdir=ROOT / "results")
            plot_pca(logits, y, f"Baseline 4-dim logits ({split} split)",
                     ROOT / f"results/figures/pca_baseline_logits_{split}.png")
            plot_pca(emb, y, f"Baseline 128-dim hidden ({split} split)",
                     ROOT / f"results/figures/pca_baseline_hidden_{split}.png")
            torch.save(model.state_dict(), ROOT / f"results/models/baseline_{split}.pt")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/metrics/baseline_runs.csv", index=False)
summary = df.groupby("split")[["train", "dev", "test", "test_song_vote"]].agg(["mean", "std"]).round(3)
summary.to_csv(ROOT / "results/metrics/baseline.csv")
print(summary)
