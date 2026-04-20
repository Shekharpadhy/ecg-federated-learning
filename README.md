# ECG Federated Learning — Explainable AI for Arrhythmia Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg-yellow)
![Explainability](https://img.shields.io/badge/XAI-SHAP%20GradientExplainer-green)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-blue)

**Privacy-Preserving Arrhythmia Classification using Federated Learning + Explainable AI**

---

## Overview

This project implements an **ECG arrhythmia classification system** that trains a shared model across **5 simulated hospitals** without ever sharing raw patient data — using **Federated Averaging (FedAvg)** — and explains every prediction using **SHAP GradientExplainer**.

The system is benchmarked in both centralised and federated settings, demonstrating that privacy and accuracy are not mutually exclusive.

> **Keywords:** `Federated Learning` · `ECG Classification` · `Explainable AI` · `Healthcare AI` · `Privacy-Preserving ML`

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What "5 Simulated Hospitals" Means](#what-5-simulated-hospitals-means)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [How It Works](#how-it-works)
6. [SHAP Explainability](#shap-explainability)
7. [ECG Feature Map](#ecg-feature-map)
8. [Code Structure](#code-structure)
9. [Setup & Usage](#setup--usage)
10. [Performance](#performance)
11. [Limitations & Future Work](#limitations--future-work)

---

## Problem Statement

Cardiovascular diseases are a leading cause of global mortality. ECG-based diagnosis is effective — but training ML models on ECG data requires access to large amounts of **sensitive patient records**, raising serious privacy, legal (HIPAA/GDPR), and security concerns.

Traditional centralised learning forces all hospitals to pool their data in one place — a major breach risk. This project solves that using **Federated Learning**: hospitals collaborate on a shared model without ever moving raw patient data off-site.

---

## What "5 Simulated Hospitals" Means

In real federated learning, each hospital owns its own patient data and trains locally. Since this project uses a single public dataset, we **simulate** this by splitting the data into 5 equal partitions — each representing one hospital's private records.

```
All ~21,849 ECG beats
        │
        ├─ Hospital 1: beats     1 –  4,370   (trains locally, no data shared)
        ├─ Hospital 2: beats  4,371 –  8,740
        ├─ Hospital 3: beats  8,741 – 13,110
        ├─ Hospital 4: beats 13,111 – 17,480
        └─ Hospital 5: beats 17,481 – 21,849
                │
                ▼  only model weights are shared (never raw ECG data)
           Global Model  ←  FedAvg (average of all 5 weight sets)
```

Each round: clients train locally → send weights to aggregator → FedAvg produces new global model → distribute back.
The experiment shows that 5 "hospitals" training in isolation can match a model trained on all the data centrally.

---

## Dataset

### MIT-BIH Arrhythmia Database (Primary)
- **Source:** [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- **Records used:** 10 records — `100, 101, 103, 105, 106, 108, 109, 111, 112, 113`
- **Total beats:** ~21,849 individual heartbeat segments
- **Beat window:** 200 samples (100 pre-peak + 100 post-peak)
- **Auto-download:** records are fetched from PhysioNet automatically on first run

### Arrhythmia Classes

| Symbol | Class | Approx. samples |
|--------|-------|-----------------|
| `N` | Normal beat | ~16,274 |
| `L` | Left bundle branch block | ~4,614 |
| `V` | Premature ventricular contraction | ~618 |
| `~` | Signal quality change | ~188 |
| `A` | Atrial premature beat | ~44 |
| `\|` | Isolated QRS-like artifact | ~42 |
| `+` | Paced beat | ~41 |
| `x` | Non-conducted P-wave (blocked APB) | ~11 |
| `Q` | Unclassifiable beat | ~7 |
| `a` | Aberrated atrial premature beat | ~6 |
| `F` | Fusion of ventricular and normal beat | ~4 |

> Classes with fewer than 2 samples are filtered automatically before training to ensure valid stratified splits.

### PTB-XL (Optional)
- Download from [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- Place at `data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
- Auto-merged with MIT-BIH if present; the project works without it

---

## Architecture

| Component | Description | Technology |
|-----------|-------------|------------|
| Data sources | MIT-BIH (10 records, ~21,849 beats) + optional PTB-XL | WFDB, NumPy |
| Preprocessing | Beat extraction, NPZ caching, rare-class filtering | NumPy, scikit-learn |
| Local model | FC network (200→64→32→N) **or** 1D CNN (3×Conv1d + AdaptiveAvgPool + Dropout) | PyTorch |
| Federated training | Manual FedAvg — 5 clients · 3 rounds · 2 local epochs · pure Python/NumPy | Python, NumPy |
| Explainability | GradientExplainer with ECG-anatomy-annotated dual-panel plot | SHAP |
| Dashboard | Live training stream, real-time per-epoch charts, upload & analyse | Streamlit |

### System Architecture

![System Architecture](results/system_architecture.png)

Data flows from MIT-BIH (and optionally PTB-XL) through preprocessing into 5 equal client partitions. Each round, hospitals train locally and send **only model weights** to the FedAvg aggregator — raw patient data never moves. After 3 rounds the global model is evaluated and explained via SHAP, with all outputs surfaced in the Streamlit dashboard.

---

## How It Works

```
1. Download MIT-BIH records (auto, first run only — via PhysioNet WFDB)
2. Extract 200-sample beat windows centred on each annotated R-peak
3. Cache processed beats to data/processed_beats.npz (near-instant on repeat runs)
4. Filter rare classes (< 2 samples) and split 80/20 stratified train/test

── Baseline mode ──────────────────────────────────────────────────────────
5a. Train FC/CNN model on full training set (7 epochs, batch=512)
5b. Per-epoch loss + accuracy logged to console in real time
5c. Evaluate → save baseline_metrics.json + baseline_model.pt

── Federated mode ─────────────────────────────────────────────────────────
5d. Partition training data equally across 5 virtual hospitals
5e. For each of 3 rounds:
      • Each hospital trains locally (2 epochs, FC/CNN model)
      • Aggregator collects all 5 weight tensors
      • FedAvg: element-wise average → new global model weights
      • Evaluate global model on shared test set → log accuracy
5f. Save federated_history.json (per-round accuracy + loss)

── Explain mode ───────────────────────────────────────────────────────────
5g. Load trained baseline model
5h. Run GradientExplainer (backprop-based SHAP) on 30 test beats
5i. Map feature indices to ECG anatomy regions (P-wave, QRS, T-wave …)
5j. Save dual-panel plot: ECG waveform overlay + top-20 feature bar chart
```

---

## SHAP Explainability

SHAP (SHapley Additive Explanations) answers: *which part of the ECG signal most influenced this prediction?*

**GradientExplainer** is used — a backpropagation-based method purpose-built for neural networks, orders of magnitude faster than the classic KernelExplainer.

### Output: dual-panel plot

**Left panel** — Mean ECG beat with SHAP importance overlaid as a colour-coded shaded region. Each anatomical segment (P-wave, QRS complex, T-wave, etc.) is highlighted. The red fill shows where importance is highest.

**Right panel** — Top-20 most important signal samples, coloured by ECG region. Labels show `S103 [QRS complex]` instead of a generic `Feature 103`.

![SHAP Plot](results/shap_summary.png)

---

## ECG Feature Map

The 200-sample beat window maps directly to standard ECG anatomy:

| Sample range | ECG region | Clinical meaning |
|---|---|---|
| 0 – 49 | Pre-beat baseline | Isoelectric reference |
| 50 – 79 | **P-wave** | Atrial depolarisation |
| 80 – 94 | PR interval | AV node conduction delay |
| **95 – 109** | **QRS complex** ⭐ | Ventricular depolarisation — highest SHAP importance |
| 110 – 139 | ST segment | Ischaemia / infarction marker |
| 140 – 169 | T-wave | Ventricular repolarisation |
| 170 – 199 | Post-beat baseline | Recovery period |

> ⭐ Sample 103 sits 3 samples after the R-peak — deep inside the QRS complex — exactly what a cardiologist examines first.

---

## Code Structure

```
ecg-federated-learning/
│
├── data/
│   ├── mit-bih-arrhythmia-database-1.0.0/   ← auto-downloaded on first run
│   └── processed_beats.npz                   ← auto-generated beat cache
│
├── results/
│   ├── baseline_metrics.json                 ← accuracy, per-epoch history, report
│   ├── baseline_model.pt                     ← saved PyTorch model weights
│   ├── federated_history.json                ← per-round federated accuracy
│   ├── shap_summary.png                      ← dual-panel SHAP plot
│   ├── shap_region_summary.json              ← SHAP importance by ECG region
│   ├── ecg_sample.png                        ← sample ECG beat visualisation
│   └── system_architecture.png              ← end-to-end pipeline diagram
│
├── src/
│   ├── config.py                ← all hyperparameters (records, epochs, clients …)
│   ├── download_data.py         ← auto PhysioNet downloader (wfdb.dl_files)
│   ├── data_utils.py            ← loading, NPZ caching, train/test split
│   ├── ptbxl_utils.py           ← optional PTB-XL loader
│   ├── model.py                 ← ECGModel (FC) + ECGCNNModel (1D CNN) + training loop
│   ├── train_baseline.py        ← centralised training, saves model + metrics
│   ├── federated_simulation.py  ← manual FedAvg loop (no Ray / no Flower)
│   └── explain.py               ← GradientExplainer + ECG-annotated dual-panel plot
│
├── dashboard.py                 ← Streamlit dashboard (6 tabs, live streaming)
├── main.py                      ← CLI entry point
└── requirements.txt
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard (recommended)
```bash
streamlit run dashboard.py
```
The dashboard handles everything: dataset download, training with live progress, SHAP explainability, and ECG upload & analyse — all in one place across 6 tabs.

### 3. CLI usage (alternative)
```bash
# Centralised baseline (7 epochs, per-epoch progress shown)
python main.py --mode baseline

# Federated learning (5 hospitals, 3 rounds)
python main.py --mode federated

# SHAP explainability on trained model
python main.py --mode explain
```

> **Note:** MIT-BIH records are downloaded automatically from PhysioNet on first run. Subsequent runs load from the local NPZ cache and are near-instant.

### 4. Model selection
Edit `src/config.py` to switch between the two model architectures:

```python
MODEL_TYPE = "fc"    # Fast fully-connected (200→64→32→N) — default
MODEL_TYPE = "cnn"   # 1D CNN (3×Conv1d + AdaptiveAvgPool + Dropout) — higher accuracy
```

---

## Performance

| Mode | Expected time | Beats used |
|---|---|---|
| Baseline | ~15–30 sec | ~21,849 |
| Federated | ~1–2 min | ~21,849 (split equally across 5 clients) |
| Explain | ~10–20 sec | 30 test samples |

> Times measured on CPU with `MODEL_TYPE = "fc"`. Switching to CNN will increase training time.

### Key findings
- Federated model accuracy is within 1–2% of the centralised baseline
- SHAP confirms the model focuses on the **QRS complex** (samples 95–109) — clinically the most informative region
- Privacy is preserved: no raw patient data ever leaves a simulated hospital partition

---

## Limitations & Future Work

| Limitation | Potential improvement |
|---|---|
| Simulated IID data split across clients | Non-IID splits to better simulate real hospital distributions |
| Single ECG lead (lead II from MIT-BIH) | Use all 12 leads from PTB-XL for richer features |
| No differential privacy | Add calibrated noise via DP-SGD |
| Fixed equal partitioning | Unequal, realistic hospital sizes |
| No secure aggregation | Implement cryptographic weight aggregation |
| Offline simulation only | Deploy on actual distributed nodes |

---

## Stack

`PyTorch` · `SHAP` · `Streamlit` · `WFDB` · `scikit-learn` · `NumPy` · `Pandas` · `Matplotlib`
