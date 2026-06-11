# Multimodal Graph Neural Network for ccRCC Survival Prediction

> Master's thesis project — Predicting 12-month survival in Clear Cell Renal Cell Carcinoma using multimodal Graph Neural Networks combining Whole Slide Images, clinical data, and genomic markers.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Dataset](#3-dataset)
4. [Methodology & Pipeline](#4-methodology--pipeline)
5. [Models](#5-models)
6. [Negative Results & Dead Ends](#6-negative-results--dead-ends)
7. [Results](#7-results)
8. [Repository Structure](#8-repository-structure)
9. [Installation & Usage](#9-installation--usage)
10. [Requirements](#10-requirements)
11. [Citation / Contact](#11-citation--contact)

---

## 1. Project Overview

This project addresses the binary classification task of **12-month survival prediction** in **Clear Cell Renal Cell Carcinoma (ccRCC)**, the most common and lethal subtype of kidney cancer.

The core contribution is a **multimodal fusion pipeline** that integrates three complementary data sources:

- **Histopathological images** — Whole Slide Images (WSI) processed as graph nodes via ResNet50 patch embeddings
- **Clinical data** — demographics, tumor staging (AJCC), and cancer history
- **Genomic markers** — somatic mutations (VHL, PBRM1, TTN) from CPTAC/TCGA profiling

These modalities are fused inside a heterogeneous **Graph Neural Network (GNN)**, where WSI patch nodes connect to patient nodes carrying clinical and genomic features. Three GNN architectures are benchmarked: **GraphSAGE**, **GAT**, and **RGCN**.

The pipeline also incorporates **Supervised Contrastive Learning (SupCon)** and **Triplet Loss** pre-training steps to improve the discriminability of WSI embeddings before fusion.

**Best result:** GraphSAGE achieves **BACC = 0.7794** and **Accuracy = 0.8667** on the held-out test set.

---

## 2. Motivation

### Why ccRCC?

Renal Cell Carcinoma accounts for approximately 3% of all adult malignancies, with the clear cell subtype (ccRCC) representing ~75% of cases and carrying the highest metastatic potential. Despite advances in targeted therapy and immunotherapy, prognosis remains highly variable and difficult to anticipate from routine clinical data alone. Early and accurate survival stratification is critical to guide treatment decisions and avoid both under- and over-treatment.

### Why Multimodal?

Each data modality captures a different aspect of tumor biology:

| Modality | What it captures |
|---|---|
| Histopathology (WSI) | Tumor morphology, microenvironment, cell architecture |
| Clinical features | Patient demographics, staging, comorbidities |
| Genomic markers | Driver mutations (VHL loss, PBRM1, TTN) linked to tumor aggressiveness |

No single modality is sufficient on its own. WSI contains rich spatial information but is high-dimensional and noisy. Clinical staging is predictive but coarse. Genomic markers are highly specific but binary. Fusion enables complementary evidence to be jointly leveraged.

### Why Graph Neural Networks?

WSI data is naturally structured as a **set of spatially related patches**, not a fixed-size vector. GNNs model this relational structure explicitly:
- Patch nodes encode local histological features
- Patient nodes aggregate both WSI evidence and structured clinical/genomic features
- Message passing enables the model to reason jointly over both modalities

---

## 3. Dataset

### Source

Patient data is derived from **TCGA-KIRC** (The Cancer Genome Atlas — Kidney Renal Clear Cell Carcinoma) and **CPTAC-CCRCC** (Clinical Proteomic Tumor Analysis Consortium), accessed through the GDC Data Portal.

### Download Links

| Dataset | Access |
|---|---|
| **TCGA-KIRC** (WSI + clinical + genomic) | [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-KIRC) |
| **CPTAC-CCRCC** (WSI + proteomics) | [CPTAC-CCRCC via Aspera](https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjU4NCIsInBhc3Njb2RlIjoiMTczZGYwYjBmMTI1Y2IxOTY0MmU2NmIyOGIzYzdlMjkwMmJjNWU1MiIsInBhY2thZ2VfaWQiOiI1ODQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=&redirected=true&authenticated=true) |
| **TCGA via GDC** (alternative access) | [GDC Portal — TCGA](https://portal.gdc.cancer.gov/) |

> **Note:** Access to TCGA/CPTAC data requires registration on the GDC portal and acceptance of the data access agreement for controlled-access datasets.

### Cohort

| Split | Patients | Survived 12+ mo. | Deceased < 12 mo. |
|---|---|---|---|
| Train | 479 | ~87% | ~13% |
| Validation | 18 | ~83% | ~17% |
| Test | 120 | ~87% | ~13% |
| **Total** | **617** | | |

> Class imbalance (~9:1 ratio) is a key challenge addressed via focal loss, oversampling, and balanced accuracy (BACC) as the primary evaluation metric.

### Modalities

**Whole Slide Images (WSI)**
- Format: `.svs` (Aperio) — pyramidal multi-resolution files
- Resolution: up to 40,000 × 30,000 pixels at level 0
- 2,134 tissue patches extracted across all patients after tissue segmentation
- Each patient: 1–7 slides (mean ≈ 3.5)

**Clinical & Genomic Data** (`medical_genomic_data/clinical+genomic_split.csv`)

| Feature group | Variables |
|---|---|
| Demographics | `gender`, `age_diag`, `race_*` (5 categories) |
| Tumor staging | `grade`, `ajcc_path_tumor_pt`, `ajcc_path_nodes_pn`, `ajcc_path_metastasis_pm`, `ajcc_path_tumor_stage` |
| Medical history | `cancer_history` |
| Genomic mutations | `VHL_mutation`, `PBRM1_mutation`, `TTN_mutation` |
| **Target** | `vital_status_12` (0 = deceased < 12 mo., 1 = survived ≥ 12 mo.) |

---

## 4. Methodology & Pipeline

The full pipeline is organized into five sequential steps:

```
WSI (.svs)
    │
    ▼
[Step 1] ResNet50 Feature Extraction
    │  2048-dim patch embeddings per WSI
    ▼
[Step 2] Mean & Max Pooling
    │  Patient-level aggregation → fixed 2048-dim vectors
    ▼
[Step 3] Supervised Contrastive Learning (SupCon)
    │  128-dim normalized embeddings with class-aware separation
    ▼
[Step 4] Triplet Loss Refinement
    │  256-dim embeddings with improved intra/inter-class distances
    ▼
[Step 5] Multimodal GNN Fusion
       Heterogeneous graph: patch nodes + patient nodes
       (WSI embeddings + clinical/genomic features)
       → Binary survival prediction
```

---

### Step 1 — Feature Extraction

**Notebook:** `Feature extraction & preparation/Features_Extraction.ipynb`

Each WSI is processed as follows:

1. **Tissue segmentation** — A low-resolution thumbnail (pyramid level 2) is converted to HSV color space. A color threshold isolates stained tissue from background/glass.
2. **Patch filtering** — An integral image accelerates the computation of tissue coverage ratio per patch. Only patches with ≥ 30% tissue are retained.
3. **Feature extraction** — Each 256×256 patch is passed through a **ResNet50** (ImageNet pre-trained, final FC layer removed) to produce a **2048-dimensional embedding**.
4. **Batch GPU inference** — Patches are processed in batches of 32 for GPU efficiency.
5. **Output** — Per-patient `.npz` files containing all patch embeddings.

```
Input:  patient.svs  (multi-gigabyte pyramidal image)
Output: patient_features.npz  → shape (N_patches, 2048)
```

---

### Step 2 — Mean & Max Pooling

**Notebook:** `Feature extraction & preparation/Mean_Max_Pooling.ipynb`

To obtain a single fixed-size patient representation from variable-length patch sets, two simple aggregation strategies are applied independently:

- **Mean pooling** — averages all patch embeddings
- **Max pooling** — takes the element-wise maximum across all patch embeddings (retains the strongest signal per dimension)

```
Input:  patient_features.npz  → (N_patches, 2048)
Output: WSI_features_MeanPooling/patient.npy  → (2048,)
        WSI_features_MaxPooling/patient.npy   → (2048,)
```

Max-pooled features are used as input to the contrastive and triplet loss models in Steps 3–4.

---

### Step 3 — Supervised Contrastive Learning (SupCon)

**Notebook:** `Feature extraction & preparation/Constractive_learning_Cleaned.ipynb`

Contrastive pre-training improves the class separability of patient embeddings before fusion.

**Architecture:** MLP projector — `2048 → 1024 → 512 → 128` with batch normalization and dropout.

**Training procedure:**
- **Loss:** Supervised Contrastive Loss (SupCon) + Cross-Entropy auxiliary loss
- **Augmentation:** Gaussian noise injection on embeddings (noise_std = 0.02, factor = 8×)
- **Constraint:** Excludes same-patient samples from contrastive pairs
- **Hyperparameter search:** Optuna with 100 trials optimizing a custom score: `0.7 × BACC + 0.3 × min_precision`

**Best Optuna configuration:**

| Hyperparameter | Value |
|---|---|
| Architecture | 2048 → 512 → 64 → 512 → 128 |
| Learning rate | 1.69e-05 |
| Batch size | 8 |
| Epochs | 12 |
| Val BACC | **0.7778** |
| Custom score | **0.7626** |

**Embedding quality (cosine silhouette score):**

| Set | Before SupCon | After SupCon |
|---|---|---|
| Train | -0.026 | **0.933** |
| Validation | 0.076 | **0.981** |
| Test | -0.003 | **0.564** |

The dramatic improvement demonstrates that SupCon successfully reorganizes the embedding space to separate survival classes. The moderate test silhouette (0.564 vs 0.933 on train) indicates a distribution shift between training and test cohorts — a common challenge in clinical WSI datasets.

---

### Step 4 — Triplet Loss

**Notebook:** `Feature extraction & preparation/Triplet_loss_model_Max.ipynb`

A second metric learning stage further refines embeddings using Triplet Loss, enforcing a margin between matched (same class) and unmatched (different class) pairs.

**Triplet mining:**
- 3,009 triplets generated from training embeddings
- Minority class (non-survivors) oversampled for balanced triplet construction

**Architecture:** MLP — `2048 → hidden → 256`, L2-normalized output.

**Configuration (grid search):**

| Parameter | Value |
|---|---|
| Embedding dim | 256 |
| Learning rate | 1e-4 |
| Margin | 0.2 |
| Dropout | 0.2 |
| Val triplet loss | **0.0755** |

**Downstream MLP classifier** (Optuna, 80 trials):
- Input: 256-dim triplet embeddings (max-pooled per patient)
- Best val BACC: **0.7778** — however test BACC drops to **0.4643**, indicating overfitting of the standalone MLP classifier, which is why GNN fusion outperforms the triplet-only approach.

---

### Step 5 — GNN Models (Multimodal Fusion)

**Notebooks:** `Experiments/Fusion_*_Final.ipynb`

The fusion step constructs a **heterogeneous graph** over all patients and their WSI patches:

- **Image nodes** (2,134 total): node features = 128-dim SupCon-projected embeddings
- **Patient nodes** (617 total): node features = clinical + genomic vector (concatenated with aggregated WSI features)
- **Edges:** image → patient (each patch is linked to its source patient)

Classification is performed on patient nodes only. Three GNN architectures are benchmarked (see [Section 5](#5-models)).

**Training strategy:**
- 5-fold cross-validation on the training set with 3 random seeds per fold
- Stability-aware objective: `0.7 × mean_CV_BACC − 0.25 × std_CV_BACC`
- Final model retrained on train + validation before test evaluation

---

## 5. Models

### GraphSAGE

**Notebook:** `Experiments/Fusion_SAGEConv_Final.ipynb`

GraphSAGE (Hamilton et al., 2017) uses neighborhood sampling with mean aggregation. The model aggregates features from neighboring patch nodes into patient representations via two SAGEConv layers.

**Best configuration:**

| Hyperparameter | Value |
|---|---|
| Hidden dim | 48 |
| Layers | 2 |
| Dropout | 0.5 |
| Aggregation | Mean |

**Test results: BACC = 0.7794 | Accuracy = 0.8667 | AUC = 0.8640**

GraphSAGE achieves the best overall performance, suggesting that simple mean aggregation over WSI patches is sufficient and robust for this task.

---

### GAT (Graph Attention Network)

**Notebook:** `Experiments/Fusion_GATConv_Final.ipynb`

GAT (Veličković et al., 2018) introduces attention weights over neighbors, allowing the model to focus on the most relevant patches for each patient.

**Best configuration:**

| Hyperparameter | Value |
|---|---|
| Hidden dim | 64 |
| Layers | 1 |
| Attention heads | 4 |
| Dropout | 0.14 |

**Test results: BACC = 0.7500 | Accuracy = 0.8417**

Despite its expressiveness, GAT does not outperform GraphSAGE, possibly due to the limited training set size (479 patients) constraining attention weight learning.

---

### RGCN (Relational Graph Convolutional Network)

**Notebook:** `Experiments/Fusion_RGCNConv_Final.ipynb`

RGCN (Schlichtkrull et al., 2018) handles multi-relational graphs. In addition to image→patient edges, a **k-NN patient-to-patient graph** is built on clinical+genomic features, adding a third relation type that connects patients with similar profiles.

**Relations:**
- `0`: image → patient (WSI patch to source patient)
- `1`: patient → image (inverse)
- `2`: patient ↔ patient (k-NN on clinical+genomic features)

**Additional features:** Focal loss for class imbalance, warmup learning rate scheduling.

**Test results: BACC = 0.7500 | Accuracy = 0.8417**

The RGCN matches GAT performance. The patient-to-patient edges from clinical k-NN do not provide a consistent improvement, suggesting that clinical features are already well-captured through the patient node features.

---

## 6. Negative Results & Dead Ends

Several research directions were explored but did not yield conclusive improvements. These are documented here for transparency and to guide future work.

### 6.1 — CLIP + LLM for WSI Description

We attempted to enrich WSI features by using a **CLIP-based model combined with a Large Language Model (LLM)** to generate textual descriptions of WSI patches, conditioned on the survival label of each patient. The idea was to add a semantic, language-grounded feature dimension to complement the visual ResNet50 embeddings. This approach did not produce conclusive results — the generated descriptions did not add discriminative signal over the existing visual features, likely due to the domain gap between natural image CLIP representations and histopathological images.

### 6.2 — Mean Pooling vs Max Pooling

All GNN experiments were also conducted using **Mean Pooling** features (`WSI_features_MeanPooling/`) as an alternative to Max Pooling. The results showed **no statistically significant difference** between the two pooling strategies across all three GNN architectures. Max Pooling was retained as the default for its slightly stronger empirical performance and its theoretical advantage in capturing the most salient patch-level signals.

### 6.3 — Vision-Language Models (VLMs) for WSI

We explored building a model based on **Vision-Language Models (VLMs)** specifically designed for medical imaging (similar to BioViL or pathology-specific LMMs). This approach was inconclusive due to the **limited number of unique patients and slides** in our cohort — VLMs require large-scale supervision to fine-tune effectively, and our dataset of 617 patients was insufficient to prevent overfitting and achieve reliable generalization.

---

## 7. Results

### Comparative Performance (Test Set)

| Model | BACC | Accuracy | AUC | Notes |
|---|---|---|---|---|
| **GraphSAGE** | **0.7794** | **0.8667** | **0.8640** | Best overall |
| GAT | 0.7500 | 0.8417 | — | 4 attention heads |
| RGCN | 0.7500 | 0.8417 | — | kNN patient graph |
| Triplet + MLP (standalone) | 0.4643 | 0.5556 | 0.7008 | Overfits on train |
| SupCon + MLP (val only) | 0.7778 | 0.7778 | 0.7284 | Val result |

> **Primary metric: Balanced Accuracy (BACC)** — chosen over raw accuracy due to severe class imbalance (~9:1). BACC = 0.5 × (recall_class0 + recall_class1).

### Key Observations

- **GraphSAGE is the best overall fusion model**, achieving BACC = 0.7794, confirming that simple mean aggregation over WSI patches is competitive with more complex attention or relational mechanisms on limited data.
- **Contrastive pre-training is essential**: raw ResNet50 embeddings have near-zero separability (silhouette ≈ −0.03); SupCon lifts this to 0.93 on training and 0.56 on test.
- **The standalone triplet + MLP classifier generalizes poorly** (val BACC = 0.78 → test BACC = 0.46), highlighting the benefit of the GNN fusion over a flat MLP.
- **Class imbalance remains a challenge**: all fusion models achieve higher recall on the majority (survivor) class. Improving recall on non-survivors (class 0) is the primary remaining bottleneck.
- **Distribution shift** between training and test cohorts is observable through the silhouette score gap (0.93 train → 0.56 test), likely due to cohort heterogeneity in multi-site TCGA/CPTAC data.

---

## 8. Repository Structure

```
WSI CCRCC/
│
├── Feature extraction & preparation/
│   ├── Features_Extraction.ipynb            # Step 1: ResNet50 patch feature extraction from SVS
│   ├── Mean_Max_Pooling.ipynb               # Step 2: Patient-level pooling aggregation
│   ├── Constractive_learning_Cleaned.ipynb  # Step 3: SupCon pre-training + Optuna search
│   ├── Triplet_loss_model_Max.ipynb         # Step 4: Triplet loss refinement + MLP classifier
│   └── distribution.ipynb                  # Data distribution analysis and label balance
│
├── Experiments/
│   ├── Fusion_SAGEConv_Final.ipynb          # Step 5: GraphSAGE fusion model (best)
│   ├── Fusion_GATConv_Final.ipynb           # Step 5: GAT fusion model
│   └── Fusion_RGCNConv_Final.ipynb          # Step 5: RGCN fusion model with kNN patient graph
│
├── medical_genomic_data/
│   └── clinical+genomic_split.csv          # 617 patients: clinical, genomic, labels, splits
│
├── features/                               # Raw ResNet50 patch embeddings (.npz), one per patient
├── WSI_features_MeanPooling/               # Mean-pooled patient embeddings (.npy)
├── WSI_features_MaxPooling/                # Max-pooled patient embeddings (.npy)
│
├── SVS original files/                     # Original WSI slides (.svs) — not tracked in git
│
├── codes/runs_supcon_step3_eval/           # SupCon evaluation outputs (metrics JSON, embeddings)
├── runs_supcon_step1_100/                  # Optuna logs for contrastive Step 1 (100 trials)
├── runs_supcon_step2_final/                # Final SupCon model trained on train+val
│
├── triplet_model_final_trainval.pth                 # Saved triplet model weights
├── final_mlp_optuna_custom_score_train_only.pth     # Saved MLP classifier weights
├── optuna_best_config_custom_score.json             # Best hyperparameter configuration
├── optuna_hyperparam_search_custom_score.csv        # Full Optuna search results (80 trials)
│
└── README.md
```

---

## 9. Installation & Usage

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: ≥ 16 GB VRAM for WSI processing)
- OpenSlide library installed at the system level

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yona1997/multimodal-ccRCC-prediction.git
cd multimodal-ccRCC-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Reproducing the Pipeline

Run the notebooks in order inside `Feature extraction & preparation/` then `Experiments/`:

**Step 1 — Feature Extraction**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Features_Extraction.ipynb"
```
> Requires `.svs` files in the configured folder (update `base_folder` inside the notebook).
> Output: `features/<patient_id>_features.npz`

**Step 2 — Pooling**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Mean_Max_Pooling.ipynb"
```
> Output: `WSI_features_MeanPooling/` and `WSI_features_MaxPooling/`

**Step 3 — Supervised Contrastive Learning**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Constractive_learning_Cleaned.ipynb"
```
> Runs Optuna (100 trials), retrains on train+val, saves 128-dim embeddings.

**Step 4 — Triplet Loss**
```bash
jupyter nbconvert --to notebook --execute \
  "Feature extraction & preparation/Triplet_loss_model_Max.ipynb"
```
> Saves `triplet_model_final_trainval.pth` and 256-dim refined embeddings.

**Step 5 — GNN Fusion**
```bash
# Best model
jupyter nbconvert --to notebook --execute "Experiments/Fusion_SAGEConv_Final.ipynb"

# Alternatives
jupyter nbconvert --to notebook --execute "Experiments/Fusion_GATConv_Final.ipynb"
jupyter nbconvert --to notebook --execute "Experiments/Fusion_RGCNConv_Final.ipynb"
```

### Data Setup

Place the clinical/genomic CSV at:
```
medical_genomic_data/clinical+genomic_split.csv
```

The CSV must contain: `case_id`, `vital_status_12`, `Split`, and all clinical/genomic feature columns described in [Section 3](#3-dataset).

---

## 10. Requirements

```
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0
openslide-python>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
optuna>=3.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
tqdm>=4.65.0
```

> **Note on torch-geometric:** Install the version matching your CUDA and PyTorch versions. Refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the correct command.

---

## 11. Citation / Contact

This work was conducted as a Master's thesis project on multimodal deep learning for computational pathology in oncology.

**Author:** Yona El Bar
**Email:** yona.elbar@gmail.com
**GitHub:** [https://github.com/yona1997/multimodal-ccRCC-prediction](https://github.com/yona1997/multimodal-ccRCC-prediction)

---

**Data sources:**

- TCGA-KIRC: [https://portal.gdc.cancer.gov/projects/TCGA-KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC)
- CPTAC-CCRCC: [https://proteomics.cancer.gov/programs/cptac](https://proteomics.cancer.gov/programs/cptac)

**Key references:**

- Hamilton W. et al. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
- Veličković P. et al. *Graph Attention Networks* (GAT). ICLR 2018.
- Schlichtkrull M. et al. *Modeling Relational Data with Graph Convolutional Networks* (RGCN). ESWC 2018.
- Khosla P. et al. *Supervised Contrastive Learning* (SupCon). NeurIPS 2020.
- Chen T. et al. *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). ICML 2020.
