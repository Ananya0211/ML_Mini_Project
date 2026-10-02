import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data import ROOT, GENRES

d = np.load(ROOT / "data/processed/gtzan4.npz")
X, y, sid, ss, sc = d["X"], d["y"], d["song_id"], d["split_song"], d["split_clip"]
print("clips:", X.shape, "| songs:", len(np.unique(sid)))

# 1. No song appears in more than one split (song-level)
tr, dv, te = (set(sid[ss == k]) for k in (0, 1, 2))
assert not (tr & dv) and not (tr & te) and not (dv & te), "LEAKAGE in song split!"
print("OK: no song overlap across train/dev/test")

# 2. Leakage exists in the clip-level split (expected, this is the paper's setup)
ctr, cte = set(sid[sc == 0]), set(sid[sc == 2])
print("songs shared by clip-split train and test:", len(ctr & cte))

# 3. Split sizes and class balance
for name, s in (("song", ss), ("clip", sc)):
    for k, n in enumerate(["train", "dev", "test"]):
        print(f"{name}-split {n}: {np.sum(s == k)} clips, per-genre {np.bincount(y[s == k], minlength=4)}")

# 4. Raw-data PCA (paper's Figure 1)
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
z = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6, 5))
for k, g in enumerate(GENRES):
    plt.scatter(z[y == k, 0], z[y == k, 1], s=4, alpha=0.4, label=g)
plt.legend(); plt.title("PCA of raw 500-dim audio")
plt.savefig(ROOT / "results/figures/pca_raw.png", dpi=150, bbox_inches="tight")
print("saved results/figures/pca_raw.png")
