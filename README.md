# Latent Feature Extraction for Musical Genres from Raw Audio

UE24CS352A Machine Learning mini-project. We reimplement and critically evaluate the *Deep Softmax Autoencoder* from "Latent Feature Extraction for Musical Genres from Raw Audio" (Sawhney, Vasavada, Wang, 2018), which learns genre embeddings directly from raw audio without hand-engineered features.

Our main changes to the original setup:
- **Song-level train/dev/test split.** The paper split randomly at clip level, so clips from the same song can appear in both train and test. We split by song and also keep the clip-level split so the two can be compared.
- **A loudness-only reference classifier**, to measure how much accuracy comes from signal amplitude alone.
- **An amplitude-normalised input option** (`rms_norm`), to test what models learn beyond loudness.

> **Status:** Part 1 (data pipeline, baseline, evaluation) is documented below. Sections marked *TODO* are filled in as the rest of the project lands.

## Team and contributions

| Member | Responsibility |
|---|---|
| Member 1 (add name) | Data pipeline, song-level split, two-layer NN baseline, evaluation module, loudness analysis |
| Member 2 (add name) | Autoencoder and Deep Softmax Autoencoder, extension *(TODO)* |
| Both | Write-up, slides, demo *(TODO)* |

## Repository structure

```
ML_Mini_Project/
├── README.md
├── requirements.txt
├── data/                  # not committed (see .gitignore)
│   ├── raw/<genre>/*.wav  # GTZAN audio, 4 genres
│   └── processed/gtzan4.npz
├── src/
│   ├── data.py            # audio loading, clipping, pooling, splits, load_split()
│   ├── check_data.py      # sanity checks + raw-data PCA figure
│   ├── baseline.py        # two-layer NN + training loop
│   ├── evaluate.py        # shared metrics, confusion matrix, song voting, PCA plots
│   ├── loudness_only.py   # one-feature reference classifier
│   └── run_baseline.py    # baseline experiment (raw and normalised, both splits, 3 seeds)
└── results/
    ├── figures/           # confusion matrices, PCA plots
    ├── metrics/           # CSV result tables
    └── models/            # saved weights (not committed)
```

## Setup

Python 3.10 was used.

```bash
git clone <repo-url> && cd ML_Mini_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies: `numpy`, `librosa`, `scikit-learn`, `torch`, `pandas`, `matplotlib`, `seaborn`.

## Data

We use the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and keep four genres: classical, jazz, metal, pop (100 songs each, 30 s each).

Place the audio so that the files sit **directly** inside each genre folder (some downloads nest an extra folder, which must be flattened):

```
data/raw/classical/classical.00000.wav ... (100 files)
data/raw/jazz/jazz.00000.wav ...
data/raw/metal/metal.00000.wav ...
data/raw/pop/pop.00000.wav ...
```

`jazz.00054.wav` is corrupted in common GTZAN copies; the pipeline skips unreadable files and prints `SKIPPED <file>`. This leaves **399 songs**.

### Preprocessing

1. Load each song with LibROSA at 22,050 Hz, mono.
2. Cut into non-overlapping 1-second clips (22,050 samples), dropping any partial clip.
3. Average-pool each clip by a factor of 44 (using the first 22,000 samples) to get a **500-dimensional** vector. No spectrogram, MFCC or other feature engineering.
4. Result: **11,968 clips**, cached in `data/processed/gtzan4.npz`.
5. At load time, inputs are standardised using **training-set** mean and standard deviation only.

### Splits

Both are stratified and use seed 42.

| Split | Train | Dev | Test | Notes |
|---|---|---|---|---|
| Song-level (default) | 8,368 | 1,800 | 1,800 | Songs are assigned to splits (70/15/15) *before* clips are used. No song appears in more than one split (asserted in `check_data.py`). |
| Clip-level | 8,377 | 1,795 | 1,796 | Random split of clips, as in the paper. All 399 songs appear in both train and test, so it leaks. |

## How to run

All commands are run from the repository root.

```bash
python3 src/data.py            # build data/processed/gtzan4.npz (a few minutes)
python3 src/check_data.py      # sanity checks, loudness stats, raw-data PCA figure
python3 src/loudness_only.py   # one-feature (loudness) reference classifier
python3 src/run_baseline.py    # two-layer NN baseline, raw + normalised, both splits, 3 seeds
```

### Interface for models (used by `evaluate.py`)

```python
from data import load_split
data = load_split("song")             # or "clip"; add rms_norm=True to remove amplitude cues
X, y, song_id = data["train"]         # also "dev", "test"; X is standardised, shape (n, 500)
```

Every model's `forward(x)` returns `(embedding, logits)`. Then `predict`, `clip_report`, `song_vote` and `plot_pca` in `src/evaluate.py` work on any model.

## Models

| Model | Description | Status |
|---|---|---|
| Two-layer NN (baseline) | 500 → 128 (tanh) → 4, cross-entropy | Done |
| Vanilla autoencoder | 500 → 256 → 192 → 128 → 64 → 128 → 192 → 256 → 500, MSE | *TODO* |
| Deep Softmax Autoencoder | Autoencoder plus classifier head 64 → 32 → 16 → 4, loss = γ·MSE + (1−γ)·CE, γ = 0.9 | *TODO* |

Baseline training: Adam, learning rate 1e-4, batch size 512, up to 300 epochs, early stopping on dev accuracy (patience 30), seeds 0, 1, 2.

## Results so far

### Baseline and loudness-only reference (test accuracy on clips, raw inputs)

Mean ± std over 3 seeds for the baseline.

| Model | Split | Train | Dev | Test | Test (song vote) |
|---|---|---|---|---|---|
| Two-layer NN | Song-level | 0.550 ± 0.021 | 0.417 ± 0.001 | 0.396 ± 0.002 | 0.456 ± 0.010 |
| Two-layer NN | Clip-level | 0.559 ± 0.004 | 0.419 ± 0.003 | 0.413 ± 0.001 | n/a |
| Loudness-only (logistic regression on log per-clip std) | Song-level | 0.594 | 0.648 | 0.631 | n/a |
| Loudness-only | Clip-level | 0.608 | 0.602 | 0.612 | n/a |

For reference, the paper reports 52.0% / 38.1% / 36.4% (train / dev / test) for its two-layer baseline, and 94.9% / 64.1% / 65.3% for its final model, using a clip-level split. Our dataset differs (11,968 clips versus their 8,000), so these are context, not an exact comparison. Chance is 25%.

*Song vote* averages softmax probabilities over all clips of a test song. The test set contains only about 60 songs, so each song is worth roughly 1.7 points and song-level numbers are noisy.

### Amplitude-normalised inputs (`rms_norm=True`)

*TODO: paste the `rms_norm = True` rows of `results/metrics/baseline.csv` here after running `run_baseline.py`.*

### Deep Softmax Autoencoder and extension

*TODO (Member 2).*

## Findings so far

- **Genres differ strongly in signal amplitude.** Mean per-clip standard deviation of the pooled signal: classical 0.0187, jazz 0.0498, metal 0.1031, pop 0.1484 (see `results/figures/pca_raw.png`, where the per-genre panels show widely different spreads around the same centre).
- **A single amplitude feature beats the baseline network.** The loudness-only classifier reaches about 61 to 63% test accuracy, versus about 40 to 41% for the two-layer NN on 500 inputs. Any model should be compared against this reference before claiming it learned genre structure. We have not yet established why the NN does not exploit this feature.
- **Baseline errors.** The baseline is biased toward predicting classical (recall about 0.76 to 0.81). Jazz is mostly predicted as classical (about 0.54 to 0.58 of jazz clips), and metal and pop are close to chance (about 0.28 to 0.30). See `results/figures/cm_baseline_*.png`.
- **Leakage effect is small for this baseline.** The clip-level split is only about 1.7 points higher than the song-level one (0.413 versus 0.396). The baseline is too weak to memorise songs, so the leakage comparison needs to be repeated for the Deep Softmax Autoencoder *(TODO)*.
- **Song voting helps.** Averaging clip predictions per song raised baseline test accuracy from 39.6% to 45.6%, with the caveat about the small test set above.
- **Baseline embeddings show no genre structure.** PCA of the 128-dim hidden layer and 4-dim logits is a single overlapping cloud (`results/figures/pca_baseline_*.png`).

## Limitations

- Average pooling by 44 acts as a crude low-pass filter, discarding most spectral detail. No MFCC or spectrogram features are used, by design.
- Only 4 genres and about 60 test songs, so accuracy differences of a few points are not reliable.
- The 3 seeds vary training randomness only; they all use the same song split.
- GTZAN has known issues (duplicate and mislabelled tracks, recording-level artefacts), so amplitude may partly reflect recording conditions rather than genre.

## Reproducibility

Seeds are fixed (split seed 42; training seeds 0, 1, 2). Raw audio, the processed `.npz` and model weights are git-ignored, so rebuild them with the commands above.

## Reference

Sawhney, A., Vasavada, V., Wang, W. *Latent Feature Extraction for Musical Genres from Raw Audio.* NIPS 2018 workshop paper. Code: https://github.com/arjunsawknee/Genre-Extraction. Dataset: Tzanetakis & Cook, *Musical Genre Classification of Audio Signals*, IEEE, 2002.
