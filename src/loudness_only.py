"""How much accuracy comes from loudness alone? Logistic regression on one feature: log(per-clip std)."""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "data/processed/gtzan4.npz")
X, y = d["X"], d["y"]
f = np.log(X.std(axis=1, keepdims=True) + 1e-8)          # (n_clips, 1)

for split in ("song", "clip"):
    s = d["split_song"] if split == "song" else d["split_clip"]
    clf = LogisticRegression(max_iter=1000).fit(f[s == 0], y[s == 0])
    accs = {n: round(float((clf.predict(f[s == k]) == y[s == k]).mean()), 3)
            for k, n in enumerate(["train", "dev", "test"])}
    print(f"loudness-only ({split} split):", accs)
