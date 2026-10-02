import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

GENRES = ["classical", "jazz", "metal", "pop"]

@torch.no_grad()
def predict(model, X):
    """Returns (embedding, logits) as numpy arrays."""
    model.eval()
    emb, logits = model(torch.tensor(X))
    return emb.numpy(), logits.numpy()

def clip_report(y, logits, name, outdir="results"):
    """Prints per-class metrics, saves a normalized confusion matrix, returns clip accuracy."""
    pred = logits.argmax(1)
    print(classification_report(y, pred, target_names=GENRES, digits=3))
    cm = confusion_matrix(y, pred, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=GENRES, yticklabels=GENRES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{name} confusion matrix")
    plt.tight_layout(); plt.savefig(Path(outdir) / "figures" / f"cm_{name}.png", dpi=150); plt.close()
    return accuracy_score(y, pred)

def song_vote(logits, y, song_id):
    """Average softmax probabilities over each song's clips -> one prediction per song."""
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    songs = np.unique(song_id)
    pred = np.array([p[song_id == s].mean(0).argmax() for s in songs])
    true = np.array([y[song_id == s][0] for s in songs])
    return accuracy_score(true, pred)

def plot_pca(emb, y, title, path):
    z = PCA(n_components=2).fit_transform(emb)
    plt.figure(figsize=(6, 5))
    for k, g in enumerate(GENRES):
        plt.scatter(z[y == k, 0], z[y == k, 1], s=6, alpha=0.5, label=g)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
