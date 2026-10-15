# Latent Feature Extraction for Musical Genres from Raw Audio

**Machine Learning Mini-Project**  
Reimplementation, critical evaluation, and loss weight ablation of the **Deep Softmax Autoencoder** from *"Latent Feature Extraction for Musical Genres from Raw Audio"* (Sawhney, Vasavada, & Wang, NeurIPS 2018 Workshop).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team & Contributions](#team--contributions)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Dataset Preparation](#dataset-preparation)
6. [How to Run](#how-to-run)
7. [Experimental Results](#experimental-results)
8. [Scientific Findings](#scientific-findings)
9. [Deliverables & Reports](#deliverables--reports)
10. [References](#references)

---

## Project Overview

Conventional Musical Genre Classification (MGC) relies on handcrafted spectral transforms (MFCCs, spectrograms). Sawhney et al. (2018) proposed classifying music directly from downsampled raw audio waveforms using a **Deep Softmax Autoencoder (DSAE)** trained with a joint reconstruction and classification objective:

$$\mathcal{L} = \gamma \cdot \text{MSE}(\text{reconstruction}, x) + (1 - \gamma) \cdot \text{CrossEntropy}(\text{logits}, y)$$

### Key Additions & Critical Investigations:
* **Song-Level Stratified Split:** The original paper split clips randomly, causing cross-clip song leakage. We implement a clean 70/15/15 song-level split to measure true out-of-sample generalization alongside the leaky clip split.
* **Loudness-Only Benchmark:** A minimal 1-feature reference model evaluating how much accuracy comes from raw signal amplitude rather than musical timbre.
* **Amplitude Normalization (`rms_norm=True`):** Unit-power scaling to evaluate what neural networks learn when volume cues are eliminated.
* **Loss Weight ($\gamma$) Sensitivity Analysis:** Systematic ablation across $\gamma \in [0.0, 1.0]$ to evaluate the regularizing effect of the autoencoder.

---

## Team and Contributions

| SRN | Member | Responsibilities |
|---|---|---|
| PES1UG24AM007 | **Abhay Natesh Bharadwaj** | Data loading & preprocessing pipeline, song-level split design, two-layer NN baseline, evaluation module, loudness reference classifier. |
| PES1UG24AM038 | **Ananya S** | Autoencoder backbone, Deep Softmax Autoencoder joint training, loss weight ($\gamma$) sensitivity analysis, determinism & seed stabilization. |
| - | **Both** | Experimental analysis, technical reports, presentation slides, and 
documentation. |

---

## Repository Structure

```text
ML_Mini_Project/
├── README.md                               # Project documentation & run guide
├── REPORT.md                               # Comprehensive technical report (full tables & logs)
├── requirements.txt                        # Python dependencies
├── data/                                   # Local audio cache (git-ignored)
│   ├── raw/<genre>/*.wav                   # Raw GTZAN audio (4 genres)
│   └── processed/gtzan4.npz                # 500-dim preprocessed clips
├── src/                                    # Source code
│   ├── data.py                             # Audio segmenting, pooling, splits, load_split()
│   ├── check_data.py                       # Split verification & volume distribution stats
│   ├── baseline.py                         # Two-layer NN baseline architecture & training
│   ├── autoencoder.py                      # Deep Softmax AE architecture & joint loss loop
│   ├── evaluate.py                         # Metrics, confusion matrix, song voting, set_seed()
│   ├── loudness_only.py                    # 1-feature volume reference classifier
│   ├── run_baseline.py                     # Baseline experiment runner (3 seeds)
│   ├── run_autoencoder.py                  # Deep Softmax AE experiment runner (3 seeds)
│   └── extension_gamma.py                  # Loss weight (gamma) ablation runner (3 seeds)
└── results/                                # Generated artifacts
    ├── README.md                           # Results catalog & figure interpretation guide
    ├── figures/                            # Confusion matrices, PCA plots, sensitivity curves
    ├── metrics/                            # CSV evaluation tables (summary & raw seeds)
    └── models/                             # Saved PyTorch model checkpoints (.pt)
```

---

## Environment Setup

Python **3.10** is recommended. All experiments use fixed random seeds for reproducibility.

### Option A: Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/Ananya0211/ML_Mini_Project.git
cd ML_Mini_Project

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Linux / macOS (Bash)

```bash
# 1. Clone repository
git clone https://github.com/Ananya0211/ML_Mini_Project.git
cd ML_Mini_Project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Environment
Run this one-liner to verify that PyTorch, LibROSA, and Scikit-Learn are installed properly:
```bash
python -c "import torch, librosa, sklearn, pandas, matplotlib; print('Environment is ready!')"
```

---

## Dataset Preparation

The project uses the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) across 4 genres: **Classical, Jazz, Metal, and Pop** (100 tracks each, 30 seconds at 22,050 Hz mono). Track `jazz.00054.wav` is corrupted in standard distributions and is skipped, leaving **399 tracks**.

### Build Dataset from Raw Audio
If rebuilding from scratch, place `.wav` files into either of the following directory layouts:
* `data/raw/<genre>/*.wav` *(e.g. `data/raw/classical/classical.00000.wav`)*
* `data/<genre>/*.wav`

Then run:
```bash
python src/data.py
```
*Processes 399 songs into 11,968 non-overlapping 1-second clips, average-pools by 44 into 500-dim vectors, and generates stratified song/clip splits.*

### Note: Reuse Preprocessed Dataset
After running `python src/data.py`, `data/processed/gtzan4.npz` is already present, for future runs we dont need to download the raw audio files. All training scripts automatically load this cached representation.

---

## How to Run

Ensure your virtual environment is active (`(.venv)` in prompt).

### Quickstart: Run Full Pipeline (End-to-End)

**In PowerShell:**
```powershell
python src/check_data.py; python src/loudness_only.py; python src/run_baseline.py; python src/run_autoencoder.py; python src/extension_gamma.py
```

**In Bash:**
```bash
python src/check_data.py && python src/loudness_only.py && python src/run_baseline.py && python src/run_autoencoder.py && python src/extension_gamma.py
```

---

### Step-by-Step Execution Guide

#### Step 1: Verify Dataset & Split Integrity
```bash
python src/check_data.py
```
* **Output:** Validates 0 song overlap in song split, displays per-genre volume standard deviations, and saves [`results/figures/pca_raw.png`](results/figures/pca_raw.png).
* **Runtime:** ~5 seconds.

#### Step 2: Run Loudness-Only Reference Classifier
```bash
python src/loudness_only.py
```
* **Output:** Evaluates logistic regression on $f = \log(\sigma_{\text{clip}})$.
* **Result:** Achieves **63.1% clip accuracy** and **73.3% song-vote accuracy** on volume alone.
* **Runtime:** ~2 seconds.

#### Step 3: Train Two-Layer Neural Network Baseline
```bash
python src/run_baseline.py
```
* **Output:** Trains baseline (500 $\to$ 128 $\to$ 4) across raw & normalised audio, song & clip splits, seeds 0, 1, 2.
* **Artifacts:** Saves [`results/metrics/baseline.csv`](results/metrics/baseline.csv), confusion matrices, PCA figures, and `.pt` models.
* **Runtime:** ~4-5 minutes.

#### Step 4: Train Deep Softmax Autoencoder
```bash
python src/run_autoencoder.py
```
* **Output:** Trains DSAE (Backbone 500 $\to$ 64 $\to$ 500, Classifier 64 $\to$ 4, $\gamma = 0.90$) across all settings.
* **Artifacts:** Saves [`results/metrics/deep_softmax.csv`](results/metrics/deep_softmax.csv), figures, and model checkpoints.
* **Runtime:** ~5-6 minutes.

#### Step 5: Run Loss Weight ($\gamma$) Sensitivity Analysis (Extension)
```bash
python src/extension_gamma.py
```
* **Output:** Sweeps $\gamma \in [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]$ on song split across 3 seeds.
* **Artifacts:** Saves [`results/metrics/gamma_ablation.csv`](results/metrics/gamma_ablation.csv) and [`results/figures/gamma_ablation_plot.png`](results/figures/gamma_ablation_plot.png).
* **Runtime:** ~3-4 minutes.

---

## Experimental Results

All numbers represent **mean ± standard deviation across 3 seeds (seeds 0, 1, 2)** with fixed PRNG initialization.

### 1. Main Model Comparison (Raw Audio, `rms_norm=False`)

| Model | Split Strategy | Train Acc | Dev Acc | Test Clip Acc | **Test Song Vote Acc** |
|---|---|---|---|---|---|
| **Loudness-Only (Logistic Reg)** | Song-level | 0.594 | 0.648 | 0.631 | **0.733** |
| **Loudness-Only** | Clip-level | 0.608 | 0.602 | 0.612 | — |
| **Two-Layer NN (Baseline)** | Song-level | 0.559 ± 0.023 | 0.413 ± 0.001 | 0.395 ± 0.006 | **0.450 ± 0.000** |
| **Two-Layer NN (Baseline)** | Clip-level | 0.643 ± 0.097 | 0.428 ± 0.010 | 0.416 ± 0.012 | — *(leaked)* |
| **Deep Softmax AE (Primary)** | **Song-level** | **0.901 ± 0.024** | **0.576 ± 0.035** | **0.564 ± 0.028** | **0.667 ± 0.017** |
| **Deep Softmax AE (Primary)** | **Clip-level** | **0.926 ± 0.004** | **0.563 ± 0.034** | **0.564 ± 0.015** | — *(leaked)* |

*Reference context: Sawhney et al. (2018) reported 36.4% test accuracy for baseline and 65.3% for autoencoder under a random clip-level split on an 8,000-clip subset.*

---

### 2. Amplitude-Normalised Inputs (`rms_norm=True`)
To test what the network learns beyond loudness, clips are normalized to unit power: $x \leftarrow x / \sqrt{\text{mean}(x^2) + \epsilon}$.

| Model | Split Strategy | Train Acc | Dev Acc | Test Clip Acc | **Test Song Vote Acc** |
|---|---|---|---|---|---|
| **Deep Softmax AE (Normalised)** | **Song-level** | 0.987 ± 0.008 | 0.327 ± 0.017 | **0.348 ± 0.013** | **0.422 ± 0.048** |
| **Deep Softmax AE (Normalised)** | **Clip-level** | 0.980 ± 0.024 | 0.356 ± 0.008 | **0.341 ± 0.008** | — *(leaked)* |

---

### 3. Loss Weight ($\gamma$) Sensitivity Analysis (Member 2 Extension)
Joint objective: $\mathcal{L} = \gamma \cdot \text{MSE} + (1 - \gamma) \cdot \text{CrossEntropy}$ on clean song split:

| $\gamma$ (Reconstruction Weight) | Loss Balance | Test Clip Acc (mean ± std) | **Test Song Vote Acc (mean ± std)** |
|---|---|---|---|
| **0.00** | Pure Classification (0% MSE, 100% CE) | 0.518 ± 0.017 | 0.650 ± 0.044 |
| **0.25** | High Classification (25% MSE, 75% CE) | 0.518 ± 0.006 | 0.667 ± 0.000 |
| **0.50** | Balanced Joint Loss (50% MSE, 50% CE) | 0.543 ± 0.020 | **0.689 ± 0.035** *(Peak)* |
| **0.75** | High Reconstruction (75% MSE, 25% CE) | 0.551 ± 0.047 | 0.650 ± 0.088 |
| **0.90** | Paper Default (90% MSE, 10% CE) | 0.564 ± 0.028 | **0.667 ± 0.017** |
| **0.95** | Dominant Reconstruction (95% MSE, 5% CE) | **0.566 ± 0.033** | 0.683 ± 0.058 |
| **1.00** | Pure Autoencoding (100% MSE, 0% CE) | 0.263 ± 0.022 | 0.250 ± 0.000 *(Random Chance)* |

*Refer REPORT.md for more details.*

---

## Scientific Findings

1. **Autoencoder Reconstruction Provides Effective Regularization:**  
   The Deep Softmax AE outperforms the 2-layer NN baseline by **+16.9% clip accuracy** (56.4% vs. 39.5%) and **+21.7% song vote accuracy** (66.7% vs. 45.0%). In the ablation, adding reconstruction loss raises song voting from 65.0% ($\gamma=0.0$) to **68.9%** ($\gamma=0.50$).
2. **The Loudness Confound:**  
   Average-pooling 22,000 samples into 500 bins eliminates fine-grained pitch, timbre, and harmonic phase. When amplitude cues are removed via RMS normalization, accuracy collapses from 56.4% down to **34.8%**. A 1-feature volume baseline reaches **63.1% clip / 73.3% song-vote accuracy**, proving that raw waveform classification primarily functions as a volume detector.
3. **Clip-Level Splits Suffer from Leakage:**  
   Random clip assignment leaks track-level mastering signatures across train and test partitions. Robust MIR benchmarks mandate strict song-level stratified partitioning.
4. **Song-Level Ensembling is Vital:**  
   Averaging clip posterior probabilities across all clips of a track improves accuracy by **+5% to +11%** across all evaluated architectures.

---

## References

1. Sawhney, A., Vasavada, V., & Wang, W. (2018). *Latent Feature Extraction for Musical Genres from Raw Audio.* NeurIPS 2018 Workshop on Machine Learning for Creativity and Design. [Code Repository](https://github.com/arjunsawknee/Genre-Extraction).
2. Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals.* IEEE Transactions on Speech and Audio Processing, 10(5), 293–302.
3. Sturm, B. L. (2014). *The state of the art ten years after a state of the art: A review of the GTZAN dataset in music information retrieval.* Journal of New Music Research, 43(2), 147–172.
