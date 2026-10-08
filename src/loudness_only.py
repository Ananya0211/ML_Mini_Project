"""How much accuracy comes from loudness alone?
Logistic regression on ONE feature: log(per-clip std). Also reports song-vote accuracy
(average class probabilities over a test song's clips) so it can be compared fairly with the neural models."""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "data/processed/gtzan4.npz")
X, y, sid = d["X"], d["y"], d["song_id"]
f = np.log(X.std(axis=1, keepdims=True) + 1e-8)          # (n_clips, 1)

def song_vote_acc(proba, y, sid):
    songs = np.unique(sid)
    pred = np.array([proba[sid == s].mean(0).argmax() for s in songs])
    true = np.array([y[sid == s][0] for s in songs])
    return float((pred == true).mean()), len(songs)

for split in ("song", "clip"):
    s = d["split_song"] if split == "song" else d["split_clip"]
    clf = LogisticRegression(max_iter=1000).fit(f[s == 0], y[s == 0])
    accs = {n: round(float((clf.predict(f[s == k]) == y[s == k]).mean()), 3)
            for k, n in enumerate(["train", "dev", "test"])}
    print(f"loudness-only ({split} split) clip accuracy:", accs)
    if split == "song":   # song voting is only meaningful when test songs are unseen
        te = s == 2
        acc, n = song_vote_acc(clf.predict_proba(f[te]), y[te], sid[te])
        print(f"loudness-only (song split) test song-vote accuracy: {acc:.3f}  ({n} test songs)")
