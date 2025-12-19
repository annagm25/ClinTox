
# ClinTox: Hybrid Approaches in Chemoinformatics for Toxicity Classification

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-Geometric-red?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)
[![App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

**Integrating Molecular Fingerprints and Geometric Deep Learning to predict clinical trial toxicity.**

---

## 📌 Project Overview

[cite_start]Drug discovery is plagued by high attrition rates, often due to unforeseen toxicity in late-stage clinical trials[cite: 372]. This project rigorously benchmarks two competing approaches for **in silico toxicity prediction** using the highly imbalanced **ClinTox** dataset ($N=1477$):

1.  **XGBoost (Gradient Boosting):** Utilizing engineered molecular descriptors (ECFP4 fingerprints, physicochemical properties).
2.  **GIN (Graph Isomorphism Network):** An end-to-end geometric deep learning architecture.

[cite_start]Contrary to the "Deep Learning superiority" assumption, our findings demonstrate that **XGBoost is the more robust and deployment-ready solution** for this specific data-scarce environment, acting as a "Sniper" (high precision) compared to GIN's "Shotgun" (high recall, low precision) approach [cite: 689-691].

## 📊 Key Results

The models were evaluated using a **Scaffold Split** (80/10/10) to test generalization to unseen chemical structures.

| Model | AUROC | Precision | Recall | Macro F1 | Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **0.957** | **0.60** | **0.86** | **0.82** | **Robust & Precise ("Sniper")** |
| **GIN** | 0.907 | 0.31 | 0.86 | 0.64 | High False Positives ("Shotgun") |

[cite_start]*Detailed metrics derived from the test set[cite: 604, 605].*

> [cite_start]**Conclusion:** While both models detect toxic compounds equally well ($Recall \approx 0.86$), XGBoost reduces false alarms by nearly 50%, making it the superior choice for automated screening pipelines[cite: 691].

## 🛠️ Methodology

### Data Pipeline
* **Source:** ClinTox dataset (via MoleculeNet).
* [cite_start]**Imbalance:** High class imbalance (only 7.6% toxic compounds)[cite: 455].
* [cite_start]**Preprocessing:** * **Tabular:** RDKit extraction of 2048-bit Morgan Fingerprints (ECFP4) + Lipinski descriptors[cite: 478].
    * **Graph:** Molecular graphs with atom/bond features for GIN.
    * [cite_start]**Splitting:** Deterministic **Scaffold Split** to prevent data leakage[cite: 520].

### Architectures
* [cite_start]**XGBoost:** Ensemble of 300 estimators with `scale_pos_weight` to handle imbalance[cite: 539, 541].
* [cite_start]**GIN:** 3-layer Graph Isomorphism Network with global addition pooling and weighted BCE loss[cite: 510, 551].

## 🚀 Installation

To replicate this study or run the app locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/annagm25/ClinTox.git](https://github.com/annagm25/ClinTox.git)
    cd ClinTox
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage: Toxicity Prediction App

[cite_start]We provide a **Streamlit** web application to easily interact with the optimized XGBoost model[cite: 666].

### Run the App
```bash
streamlit run streamlit/app.py
