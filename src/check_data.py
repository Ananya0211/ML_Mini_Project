import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data import ROOT, GENRES

d = np.load(ROOT / "data/processed/gtzan4.npz")
X, y, sid, ss, sc = d["X"], d["y"], d["song_id"], d["split_song"], d["split_clip"]
print("clips:", X.shape, "| songs:", len(np.unique(sid)))

# 1. No song appears in more than one split (song-level split)
tr, dv, te = (set(sid[ss == k]) for k in (0, 1, 2))
assert not (tr & dv) and not (tr & te) and not (dv & te), "LEAKAGE in song split!"
print("OK: no song overlap across train/dev/test (song split)")

# 2. The clip-level split leaks songs (this is the paper's setup)
print("songs shared by clip-split train and test:", len(set(sid[sc == 0]) & set(sid[sc == 2])))

# 3. Split sizes and class balance
for name, s in (("song", ss), ("clip", sc)):
    for k, n in enumerate(["train", "dev", "test"]):
        print(f"{name}-split {n}: {np.sum(s == k)} clips, per-genre {np.bincount(y[s == k], minlength=4)}")

# 4. Raw-data PCA (paper's Figure 1): combined plot + one panel per genre
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
z = PCA(n_components=2).fit_transform(X)
rng = np.random.default_rng(0)
order = rng.permutation(len(y))                      # shuffle so no genre is always drawn on top
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
ax[0].scatter(z[order, 0], z[order, 1], c=[colors[k] for k in y[order]], s=3, alpha=0.4)
ax[0].set_title("All genres")
for k, g in enumerate(GENRES):
    ax[k + 1].scatter(z[y == k, 0], z[y == k, 1], s=3, alpha=0.4, color=colors[k])
    ax[k + 1].set_title(g)
plt.tight_layout()
plt.savefig(ROOT / "results/figures/pca_raw.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved results/figures/pca_raw.png")
for k, g in enumerate(GENRES):
    print(g, "mean per-clip std:", X[y == k].std(axis=1).mean().round(4))
