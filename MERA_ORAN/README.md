---
editor_options: 
  markdown: 
    wrap: 72
---

# Manifold-Optimized MERA Wavelets for O-RAN Fronthaul Compression

**A Data-Driven Framework for "Learning to Compress" 5G Signals**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-orange.svg)
![Status](https://img.shields.io/badge/status-Paper_Ready-green.svg)

## 📖 Overview

This repository houses the implementation of a novel compression scheme
for **5G/6G Fronthaul links (O-RAN)**. It addresses the massive
bandwidth bottleneck between Radio Units (RU) and Distributed Units (DU)
by replacing standard scalar quantization (CPRI) with an **Adaptive
Unitary Filter Bank**.

### The Core Idea

Standard methods like CPRI or classical Wavelets (Daubechies) use
**fixed** bases. They treat the 5G signal as generic noise. Our method
uses **MERA (Multiscale Entanglement Renormalization Ansatz)**
structures to **learn** the optimal basis for the specific traffic load
of the network.

By optimizing filters directly on the **Riemannian Manifold of Unitary
Matrices** $U(M)$, we achieve: 1. **Perfect Reconstruction (PR)** by
structural design. 2. **Maximum Sparsity** adapted to the instantaneous
traffic load.

------------------------------------------------------------------------

## 🚀 Key Results

Experiments simulating a **30% Traffic Load** scenario (typical daily
usage) demonstrate significant gains over the industry standard:

| Metric | Standard CPRI (4 bits) | **MERA K=2 (4 bits)** | Impact |
|:-----------------|:-----------------|:-----------------|:-----------------|
| **EVM** | 12.7% (High Distortion) | **\~7.5%** (3GPP Compliant) | Enables **64-QAM** transmission |
| **SNR** | 17.9 dB | **22.5 dB** | **+4.6 dB Coding Gain** |
| **Constellation** | Noisy / Error Floor | Clean / Distinct Symbols | Reduces burden on FEC |

> **Visual Proof:** See `results/constellation_comparison.png` after
> running the experiment.

------------------------------------------------------------------------

## 📂 Repository Structure

The code is organized into MATLAB packages (namespaces) to ensure
modularity.

fronthaul_wavelets/
├── main_experiment.m            # 🚀 START HERE: Runs the full Rate-Distortion loop.
├── plot_spectral_analysis.m     # Validation: Proves the "Band-Limited" nature of traffic.
├── spectral_modeling_analysis.m # Theory: Fits Yule-Walker AR models to the spectrum.
│
├── +mera/                       # The Core Library
│   ├── FilterBank.m             # Implements the Unitary Lattice Factorization.
│   └── train.m                  # Manifold Optimization (Riemannian Gradient Descent).
│
├── +signal/                     # Signal Generation
│   └── generate_5g_iq.m         # Generates 5G NR-compliant waveforms (Manual implementation).
│
├── +quantizer/                  # Bit Loading & Quantization
│   ├── allocate_bits.m          # Greedy algorithm for dynamic bit allocation.
│   ├── uniform.m                # Forward Quantizer.
│   └── uniform_inverse.m        # Inverse Quantizer.
│
├── +baselines/                  # Benchmarking
│   ├── cpri_standard.m          # Standard scalar quantization.
│   └── daubechies_compress.m    # Standard wavelet transform.
│
├── +metrics/                    # Evaluation
│   └── compute_evm.m            # Error Vector Magnitude (3GPP definition).
│
└── results/                     # Output directory for plots and stats.
