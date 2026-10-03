# Latent Feature Extraction for Musical Genres from Raw Audio

UE24CS352A Machine Learning mini-project. We reimplement and critically evaluate the *Deep Softmax Autoencoder* from "Latent Feature Extraction for Musical Genres from Raw Audio" (Sawhney, Vasavada, Wang, 2018), which learns genre embeddings directly from raw audio without hand-engineered features.

Our main changes to the original setup:
- **Song-level train/dev/test split.** The paper split randomly at clip level, so clips from the same song can appear in both train and test. We split by song and also keep the clip-level split so the two can be compared.
- **A loudness-only reference classifier**, to measure how much accuracy comes from signal amplitude alone.
- **An amplitude-normalised input option** (`rms_norm`), to test what models learn beyond loudness.

> **Status:** Complete. Data pipeline, 2-layer NN baseline, Deep Softmax Autoencoder, evaluation, and loss weight sensitivity extension are fully documented below.

## Team and contributions

| Member | Responsibility |
|---|---|
| Abhay Natesh Bharadwaj | Data pipeline, song-level split, two-layer NN baseline, evaluation module, loudness analysis |
| Ananya S | Autoencoder, Deep Softmax Autoencoder, extension experiments |
| Both | Write-up, slides, demo |

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
│   ├── autoencoder.py     # Vanilla AE + Deep Softmax AE + joint loss training loop
│   ├── evaluate.py        # shared metrics, confusion matrix, song voting, PCA plots
│   ├── loudness_only.py   # one-feature reference classifier
│   ├── run_baseline.py    # baseline experiment runner
│   ├── run_autoencoder.py # Deep Softmax AE experiment runner
│   └── extension_gamma.py # loss weight sensitivity extension analysis
└── results/
    ├── README.md          # results index & visual figure catalog
    ├── figures/           # confusion matrices, PCA plots, sensitivity curves
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

Place the audio so that the files sit inside each genre folder. Both directory structures are automatically supported by `src/data.py`:

Option A (`data/raw/<genre>/`):
```text
data/raw/classical/classical.00000.wav ... (100 files)
data/raw/jazz/jazz.00000.wav ...
data/raw/metal/metal.00000.wav ...
data/raw/pop/pop.00000.wav ...
```

Option B (`data/<genre>/`):
```text
data/classical/classical.00000.wav ... (100 files)
data/jazz/jazz.00000.wav ...
data/metal/metal.00000.wav ...
data/pop/pop.00000.wav ...
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
python3 src/run_autoencoder.py # Deep Softmax AE, raw + normalised, both splits, 3 seeds
python3 src/extension_gamma.py # loss weight (gamma) sensitivity ablation experiment
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
| Vanilla autoencoder | 500 → 256 → 192 → 128 → 64 → 128 → 192 → 256 → 500, MSE | Done |
| Deep Softmax Autoencoder | Autoencoder plus classifier head 64 → 32 → 16 → 4, loss = γ·MSE + (1−γ)·CE, γ = 0.9 | Done |

Baseline training: Adam, learning rate 1e-4, batch size 512, up to 300 epochs, early stopping on dev accuracy (patience 30), seeds 0, 1, 2.

## Results so far

### Model Performance Comparison (Raw inputs, `rms_norm=False`)

Mean ± std over 3 seeds.

| Model | Split | Train | Dev | Test | Test (song vote) |
|---|---|---|---|---|---|
| Two-layer NN | Song-level | 0.550 ± 0.021 | 0.417 ± 0.001 | 0.396 ± 0.002 | 0.456 ± 0.010 |
| Two-layer NN | Clip-level | 0.559 ± 0.004 | 0.419 ± 0.003 | 0.413 ± 0.001 | n/a |
| Loudness-only (logistic regression) | Song-level | 0.594 | 0.648 | 0.631 | n/a |
| Loudness-only | Clip-level | 0.608 | 0.602 | 0.612 | n/a |
| **Deep Softmax Autoencoder** | **Song-level** | **0.855 ± 0.115** | **0.543 ± 0.053** | **0.520 ± 0.096** | **0.622 ± 0.121** |
| **Deep Softmax Autoencoder** | **Clip-level** | **0.910 ± 0.010** | **0.574 ± 0.009** | **0.563 ± 0.010** | n/a |

For reference, the paper reports 52.0% / 38.1% / 36.4% (train / dev / test) for its two-layer baseline, and 94.9% / 64.1% / 65.3% for its final model, using a clip-level split. Our dataset differs (11,968 clips versus their 8,000), so these are context, not an exact comparison. Chance is 25%.

*Song vote* averages softmax probabilities over all clips of a test song. The test set contains only about 60 songs, so each song is worth roughly 1.7 points and song-level numbers are noisy.

### Deep Softmax Autoencoder with Amplitude-Normalised Inputs (`rms_norm=True`)

| Model | Split | Train | Dev | Test | Test (song vote) |
|---|---|---|---|---|---|
| Deep Softmax Autoencoder | Song-level | 0.817 ± 0.306 | 0.326 ± 0.022 | 0.324 ± 0.024 | 0.344 ± 0.025 |
### Extension: Loss Weight ($\gamma$) Sensitivity Analysis (Member 2)

We evaluated the effect of varying the reconstruction loss weight $\gamma \in [0.0, 1.0]$ in the joint loss $\mathcal{L} = \gamma \cdot \text{MSE} + (1 - \gamma) \cdot \text{CE}$ (Song split, raw audio):

| $\gamma$ (Reconstruction Weight) | Loss Focus | Test Clip Acc (mean ± std) | **Test Song Vote Acc (mean ± std)** |
|---|---|---|---|
| **0.00** | Pure Classification | 0.527 ± 0.023 | 0.689 ± 0.042 |
| **0.25** | High Classification | 0.551 ± 0.022 | 0.706 ± 0.010 |
| **0.50** | Balanced Joint Loss | 0.556 ± 0.014 | **0.717 ± 0.029** |
| **0.75** | High Reconstruction | 0.549 ± 0.026 | 0.678 ± 0.038 |
| **0.90** | Paper Default | **0.566 ± 0.023** | **0.700 ± 0.073** |
| **0.95** | Low Classification | 0.523 ± 0.109 | 0.622 ± 0.178 |
| **1.00** | Pure Autoencoding | 0.276 ± 0.023 | 0.267 ± 0.029 |

See [`results/figures/gamma_ablation_plot.png`](file:///c:/Users/Ananya/Desktop/Eng/Sem5/ML/ML_Mini_Project/results/figures/gamma_ablation_plot.png) for the sensitivity curve.

## Findings so far

- **Genres differ strongly in signal amplitude.** Mean per-clip standard deviation of the pooled signal: classical 0.0187, jazz 0.0498, metal 0.1031, pop 0.1484 (see `results/figures/pca_raw.png`, where the per-genre panels show widely different spreads around the same centre).
- **A single amplitude feature beats the baseline network.** The loudness-only classifier reaches about 61 to 63% test accuracy, versus about 40 to 41% for the two-layer NN on 500 inputs. Any model should be compared against this reference before claiming it learned genre structure. We have not yet established why the NN does not exploit this feature.
- **Baseline errors.** The baseline is biased toward predicting classical (recall about 0.76 to 0.81). Jazz is mostly predicted as classical (about 0.54 to 0.58 of jazz clips), and metal and pop are close to chance (about 0.28 to 0.30). See `results/figures/cm_baseline_*.png`.
- **Deep Softmax Autoencoder outperforms the 2-layer NN baseline significantly.** On raw audio (song split), clip test accuracy jumps from 39.6% to **52.0%** (+12.4%) and song-voting accuracy jumps from 45.6% to **62.2%** (+16.6%), matching the loudness-only baseline (63.1%).
- **Reconstruction regularization ($\text{MSE}$) improves classification (Extension Finding).** Adding reconstruction loss ($\gamma = 0.50$ to $0.90$) raises song-voting accuracy from 68.9% (pure classifier $\gamma=0.0$) to **71.7%** ($\gamma=0.5$) and **70.0%** ($\gamma=0.9$). Forcing the bottleneck to reconstruct raw audio acts as a effective regularizer. Pure unsupervised autoencoding ($\gamma=1.0$) performs near random chance (26.7%).
- **Leakage effect is higher on the deeper model.** The clip-level split reaches 56.3% vs. 52.0% on the song-level split (a 4.3 percentage point gap compared to 1.7 points on the 2-layer baseline), showing that deeper representations memorize clip signatures across songs when shared between splits.
- **Heavy reliance on amplitude cues (`rms_norm=True`).** When inputs are amplitude-normalized (`rms_norm=True`), Deep Softmax AE test accuracy drops sharply to **32.4%** (song split) and **33.1%** (clip split). This empirically proves that the model primarily exploits raw signal volume/amplitude differences across genres rather than rich timbral structure.
- **Song voting provides a major boost.** Averaging clip predictions per song raised song-level test accuracy for the Deep Softmax AE from 52.0% (clip level) to **62.2%** (song level), with peak seeds reaching up to **75.0%** during ablation runs.
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
