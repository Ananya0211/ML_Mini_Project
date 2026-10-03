import numpy as np, librosa
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

GENRES = ["classical", "jazz", "metal", "pop"]
SR, CLIP_LEN, POOL = 22050, 22050, 44      # 500 * 44 = 22,000 samples kept per clip

def load_song(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = len(y) // CLIP_LEN
    clips = y[: n * CLIP_LEN].reshape(n, CLIP_LEN)[:, :500 * POOL]
    return clips.reshape(n, 500, POOL).mean(axis=2)           # (n_clips, 500)

def build(raw_dir=None, out=ROOT / "data/processed/gtzan4.npz", seed=42):
    X, y, sid, names, song_labels = [], [], [], [], []
    for label, g in enumerate(GENRES):
        if raw_dir is not None:
            files = sorted(Path(raw_dir, g).glob("*.wav"))
        else:
            # Check both data/raw/<genre>/*.wav and data/<genre>/*.wav
            files = sorted((ROOT / "data/raw" / g).glob("*.wav"))
            if not files:
                files = sorted((ROOT / "data" / g).glob("*.wav"))

        for p in files:
            try:
                f = load_song(p)
            except Exception as e:
                print("SKIPPED", p.name, e); continue
            s = len(names); names.append(p.name); song_labels.append(label)
            X.append(f); y += [label] * len(f); sid += [s] * len(f)

    if not X:
        raise FileNotFoundError(f"No audio files found. Check data/raw/<genre>/*.wav or data/<genre>/*.wav")

    X, y, sid = np.concatenate(X).astype(np.float32), np.array(y), np.array(sid)
    song_labels = np.array(song_labels)

    # --- song-level split (70/15/15, stratified) ---
    ids = np.arange(len(names))
    tr, tmp = train_test_split(ids, test_size=0.30, stratify=song_labels, random_state=seed)
    dv, te = train_test_split(tmp, test_size=0.50, stratify=song_labels[tmp], random_state=seed)
    song_split = np.zeros(len(names), dtype=int); song_split[dv] = 1; song_split[te] = 2
    split_song = song_split[sid]                              # per-clip: 0 train, 1 dev, 2 test

    # --- random clip-level split (reproduces the paper's leaky setup) ---
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    c, d = train_test_split(b, test_size=0.50, stratify=y[b], random_state=seed)
    split_clip = np.zeros(len(y), dtype=int); split_clip[c] = 1; split_clip[d] = 2

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, song_id=sid, split_song=split_song,
                        split_clip=split_clip, song_names=np.array(names))
    print("clips:", X.shape, "songs:", len(names))

def load_split(split="song", path=ROOT / "data/processed/gtzan4.npz", rms_norm=False):
    """Returns dict with train/dev/test -> (X, y, song_id). X standardized with TRAIN stats only.
    rms_norm=True first scales every clip to unit std, removing amplitude/loudness as a cue."""
    d = np.load(path)
    X = d["X"]
    if rms_norm:
        X = X / (X.std(axis=1, keepdims=True) + 1e-8)
    s = d["split_song"] if split == "song" else d["split_clip"]
    mu = X[s == 0].mean(axis=0); sd = X[s == 0].std(axis=0) + 1e-8
    out = {}
    for k, name in enumerate(["train", "dev", "test"]):
        m = s == k
        out[name] = ((X[m] - mu) / sd, d["y"][m], d["song_id"][m])
    return out

if __name__ == "__main__":
    build()
